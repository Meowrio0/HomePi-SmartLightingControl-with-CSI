'use strict';

const { Gpio } = require('pigpio');
const mqtt = require('mqtt');
const { spawn } = require('child_process');
const path =require('path');
const readline = require('readline');

let Service, Characteristic;

module.exports = (api) => {
    Service = api.hap.Service;
    Characteristic = api.hap.Characteristic;
    api.registerAccessory('SmartLightCSI', SmartLightCSI);
};

class SmartLightCSI {
    constructor(log, config) {
        this.log = log;
        this.cfg = config;

        this.name = config.name || 'Smart CSI Light';
        this.relayPinNum = config.gpioPin;
        this.ldrPinNum = config.ldrPin;
        this.pirPinNum = config.pirPin;
        this.usePIR = config.usePIR !== false; 
        this.invertRelay = !!config.invert;

        this.scanIntervalMs = (config.scanInterval || 5) * 1000;
        this.turnOffDelayMs = (config.turnOffDelay || config.csiNoPresenceTurnOffDelay || 30) * 1000; 
        this.manualOverrideTimeoutMs = (config.manualOverrideTimeoutMinutes || 30) * 60 * 1000;
        this.pirActiveWindowMs = (config.pirActiveWindowSeconds || 10) * 1000; 

        this.mqttUrl = config.mqttUrl || 'mqtt://127.0.0.1:1883';
        this.mqttCsiTopic = config.mqttCsiTopic || 'homebridge/csi/raw_data';
        this.csiDataTimeoutMs = (config.csiDataTimeoutSeconds || 20) * 1000;

        this.predictScriptPath = config.predictScriptPath || path.join(__dirname, 'python', 'predict.py');
        this.log.info(`[Setup] Path to predict.py: ${this.predictScriptPath}`);
        
        this.pythonExecutable = config.pythonExecutablePath || '/home/linjianxun/homebridge_py_venv/bin/python3';
        this.log.info(`[Setup] Python executable: ${this.pythonExecutable}`);

        this.lightOn = false;
        this.automationEnabled = config.automationEnabled !== false;
        
        this.pirLastActivationTimestamp = 0; 
        
        this.csiPresenceDetected = false;
        this.lastCsiDataTime = 0;
        this.turnOffTimer = null;
        
        this.manualOverrideActive = false;
        this.manualOverrideTimer = null;

        this.pythonProcess = null; 
        this.pythonRestartAttempts = 0;
        this.maxPythonRestartAttempts = 5; 

        this.informationService = new Service.AccessoryInformation()
            .setCharacteristic(Characteristic.Manufacturer, "YourName/Linjianxun")
            .setCharacteristic(Characteristic.Model, "CSI Presence Light")
            .setCharacteristic(Characteristic.SerialNumber, `GPIO-${this.relayPinNum || 'N/A'}`);

        this.lightService = new Service.Lightbulb(this.name);
        this.lightService.getCharacteristic(Characteristic.On)
            .onGet(() => this.lightOn)
            .onSet(this.setLightManually.bind(this));

        this.automationSwitchService = new Service.Switch(this.name + " Automation", "automationSwitch");
        this.automationSwitchService.getCharacteristic(Characteristic.On)
            .onGet(() => this.automationEnabled)
            .onSet((value, callback) => {
                this.log.info(`[Automation] onSet - Received value: ${value}. Current: ${this.automationEnabled}`);
                try {
                    this.automationEnabled = value;
                    this.log.info(`[Automation] automationEnabled set to: ${this.automationEnabled}`);
                    if (!this.automationEnabled) {
                        this.clearTimersAndStates(); 
                        this.log.info('[Automation] 自動化已禁用。');
                    } else {
                        this.clearManualOverride();
                        this.log.info('[Automation] 自動化已啟用。');
                        this.evaluateLightState(); 
                    }
                    if (typeof callback === 'function') callback(null);
                    else this.log.error(`[Automation] onSet - callback is not a function! Type: ${typeof callback}`);
                } catch (e) {
                    this.log.error(`[Automation] onSet error: ${e.message} ${e.stack}`);
                    if (typeof callback === 'function') callback(e);
                    else this.log.error(`[Automation] onSet error handling - callback is not a function!`);
                }
            });
        
        setTimeout(() => {
            this.initialiseGPIO();
            this.startPythonProcess(); 
            this.initialiseMQTT(); 
        }, 2000); 
    }

    startPythonProcess() {
        if (this.pythonProcess) { 
            this.log.warn('[PythonService] Existing Python process found. Terminating before restart.');
            this.pythonProcess.kill('SIGTERM'); 
            this.pythonProcess.removeAllListeners(); 
            this.pythonProcess = null;
        }

        this.log.info(`[PythonService] Spawning Python script: ${this.pythonExecutable} ${this.predictScriptPath}`);
        try {
            this.pythonProcess = spawn(this.pythonExecutable, 
                                       [this.predictScriptPath], 
                                       { stdio: ['pipe', 'pipe', 'pipe'] }); 
        } catch (spawnError) {
            this.log.error(`[PythonService] Error spawning Python process: ${spawnError.message} ${spawnError.stack}`);
            this.pythonProcess = null;
            this.handlePythonProcessCrash(); 
            return; 
        }
        
        const rl = readline.createInterface({ input: this.pythonProcess.stdout });

        rl.on('line', (line) => {
            const output = line.trim().toLowerCase();
            if (output !== 'buffering') { 
                this.log.info(`[PythonService] Received from predict.py stdout: '${output}'`);
            } else {
                this.log.debug(`[PythonService] Received from predict.py stdout: '${output}'`);
            }

            let csiStateChanged = false;
            if (output === '0') {
                if (this.csiPresenceDetected !== false) {
                    this.log.info(`[CSI Model] 預測狀態改變: ${this.csiPresenceDetected ? "之前有人" : "之前無人"} -> 現在無人`);
                    csiStateChanged = true;
                }
                this.csiPresenceDetected = false;
            } else if (output === '1') {
                if (this.csiPresenceDetected !== true) {
                    this.log.info(`[CSI Model] 預測狀態改變: ${this.csiPresenceDetected ? "之前有人" : "之前無人"} -> 現在有人`);
                    csiStateChanged = true;
                }
                this.csiPresenceDetected = true;
            } else if (output === 'buffering') {
                // Buffering, do not change csiPresenceDetected, do not call evaluateLightState yet
                return; 
            } else if (output === 'error' || output === 'critical_error_exit') {
                this.log.error(`[PythonService] predict.py signaled an error or critical exit via stdout: '${output}'`);
                if (this.csiPresenceDetected !== false) { 
                    this.log.info(`[CSI Model] Python 腳本錯誤，預測狀態保守設為: 之前有人 -> 現在無人`);
                    csiStateChanged = true; 
                }
                this.csiPresenceDetected = false; 
            } else { 
                this.log.warn(`[PythonService] Unknown output from predict.py: '${output}'. Defaulting to no presence.`);
                if (this.csiPresenceDetected !== false) {
                    this.log.info(`[CSI Model] Python 腳本未知輸出，預測狀態保守設為: 之前有人 -> 現在無人`);
                    csiStateChanged = true;
                }
                this.csiPresenceDetected = false;
            }
            
            // Only evaluate light state if CSI state potentially changed to a non-buffering state,
            // or if it's an error that implies a state change.
            if (csiStateChanged || output === 'error' || output === 'critical_error_exit') {
                this.evaluateLightState();
            }
        });

        this.pythonProcess.stderr.on('data', (data) => {
            const stderrStr = data.toString().trim();
            if (stderrStr) { 
                if (stderrStr.includes("InconsistentVersionWarning") || 
                    stderrStr.includes("tf.lite.Interpreter is deprecated") ||
                    stderrStr.includes("TensorFlow Lite delegate") ||
                    stderrStr.includes("NodeDef mentions attribute use_inter_op_parallelism")) {
                    this.log.debug(`[predict.py stderr - KnownWarning] ${stderrStr}`);
                } else {
                    this.log.error(`[predict.py stderr] ${stderrStr}`);
                }
            }
        });

        this.pythonProcess.on('error', (err) => {
            this.log.error(`[PythonService] Failed to start Python process or runtime error: ${err.message}`);
            this.handlePythonProcessCrash();
        });

        this.pythonProcess.on('close', (code, signal) => {
            this.log.warn(`[PythonService] Python process exited with code ${code} and signal ${signal}`);
            const wasKilledByUs = this.pythonProcess?.killed; // Check before nulling
            this.pythonProcess = null; 
            
            if (!wasKilledByUs && code !== 0 && code !== null) { 
                 this.handlePythonProcessCrash();
            } else {
                this.log.info(`[PythonService] Python process exited cleanly (code ${code}, signal ${signal}) or was killed by plugin. Resetting restart attempts.`);
                this.pythonRestartAttempts = 0; 
            }
        });
        this.log.info('[PythonService] Python process spawned and listeners configured.');
    }

    handlePythonProcessCrash() {
        if (this.pythonProcess) { 
             this.pythonProcess.removeAllListeners();
             this.pythonProcess = null;
        }
        this.pythonRestartAttempts++;
        if (this.pythonRestartAttempts <= this.maxPythonRestartAttempts) {
            this.log.info(`[PythonService] Python process crashed/exited unexpectedly. Attempting to restart in 5 seconds (attempt ${this.pythonRestartAttempts}/${this.maxPythonRestartAttempts}).`);
            setTimeout(() => this.startPythonProcess(), 5000);
        } else {
            this.log.error(`[PythonService] Max restart attempts reached for Python process. Will not restart again until Homebridge restarts or plugin reloads.`);
        }
    }

    initialiseMQTT() {
        try {
            this.mqttClient = mqtt.connect(this.mqttUrl, { reconnectPeriod: 5000, connectTimeout: 10000 });
            this.mqttClient.on('connect', () => {
                this.log.info('[MQTT] 已連線 →', this.mqttUrl);
                this.mqttClient.subscribe(this.mqttCsiTopic, (err) => {
                    if (err) this.log.error(`[MQTT] 訂閱主題 ${this.mqttCsiTopic} 失敗：`, err.message);
                    else this.log.info('[MQTT] 成功訂閱主題：', this.mqttCsiTopic);
                });
            });
            this.mqttClient.on('reconnect', () => this.log.debug('[MQTT] 重連中…'));
            this.mqttClient.on('error', (e) => this.log.error('[MQTT] 錯誤：', e.message));
            this.mqttClient.on('offline', () => this.log.warn('[MQTT] 離線'));

            this.mqttClient.on('message', (topic, payload) => {
                this.log.debug(`[MQTT Message] Received on topic: ${topic}`);
                try {
                    if (topic !== this.mqttCsiTopic) return;
                    this.lastCsiDataTime = Date.now();
                    
                    let csiPacket;
                    try {
                        csiPacket = JSON.parse(payload.toString());
                        if (!csiPacket.csi_raw || typeof csiPacket.rssi === 'undefined' || typeof csiPacket.mcs === 'undefined') {
                            this.log.warn('[MQTT] 收到的CSI數據包不完整或格式錯誤:', payload.toString());
                            return;
                        }
                    } catch (err) {
                        this.log.warn('[MQTT] 解析CSI JSON數據失敗:', err.message, payload.toString());
                        return;
                    }

                    const dataToSend = JSON.stringify(csiPacket); 
                    
                    if (this.pythonProcess && this.pythonProcess.stdin && !this.pythonProcess.stdin.destroyed) {
                        this.log.debug(`[PythonService] Sending to python stdin: ${dataToSend.substring(0,100)}...`);
                        try {
                             this.pythonProcess.stdin.write(dataToSend + '\n');
                        } catch (e) {
                            this.log.error(`[PythonService] Error writing to python stdin: ${e.message}. Process might be closing or crashed.`);
                        }
                    } else {
                        this.log.warn(`[PythonService] Python process not running or stdin not available when MQTT message received. CSI data for this message dropped. Waiting for automatic restart if configured.`);
                        if (!this.pythonProcess && this.pythonRestartAttempts <= this.maxPythonRestartAttempts) { // Check attempts to avoid loop if maxed out
                             this.log.info(`[PythonService] MQTT message received but process is down. Triggering a restart check (if attempts remain).`);
                             this.handlePythonProcessCrash(); 
                        }
                    }
                } catch (messageError) {
                    this.log.error(`[MQTT] Error processing message: ${messageError.message} ${messageError.stack}`);
                }
            });
        } catch (initError) {
            this.log.error(`[MQTT] 初始化失敗: ${initError.message} ${initError.stack}`);
        }
    }
    
    setLightManually(value, callback) {
        this.log.debug(`[HomeKit] 收到手動/場景設置燈狀態請求 -> ${value ? '開' : '關'}`);
        try {
            this.manualOverrideActive = true;
            if (this.turnOffTimer) { // 手動操作時，取消自動關燈計時器
                clearTimeout(this.turnOffTimer);
                this.turnOffTimer = null;
                this.log.info('[Control] 手動操作，取消自動關燈計時器。');
            }
            if (this.manualOverrideTimer) clearTimeout(this.manualOverrideTimer);
            this.manualOverrideTimer = setTimeout(() => {
                this.log.info(`[Control] 手動/場景覆蓋超時 (${this.manualOverrideTimeoutMs / 60000}分鐘)，恢復自動化邏輯。`);
                this.clearManualOverride(); // 會調用 evaluateLightState
            }, this.manualOverrideTimeoutMs);
            this.setLightState(value, "[HomeKit Set]");
            if (typeof callback === 'function') callback(null);
            else this.log.error(`[HomeKit] setLightManually - Error: callback is not a function! Type: ${typeof callback}`);
        } catch (error) {
            this.log.error(`[HomeKit] setLightManually - Error: ${error.message} ${error.stack}`);
            if (typeof callback === 'function') callback(error);
            else this.log.error(`[HomeKit] setLightManually - Error during error handling: callback is not a function!`);
        }
    }

    clearManualOverride() {
        this.manualOverrideActive = false;
        if (this.manualOverrideTimer) {
            clearTimeout(this.manualOverrideTimer);
            this.manualOverrideTimer = null;
        }
        this.log.info('[Control] 解除手動/場景覆蓋模式，將重新評估燈光狀態。');
        this.evaluateLightState(); // 恢復自動化後立即評估一次
    }
    
    clearTimersAndStates() { // 主要用於自動化禁用時
        if (this.turnOffTimer) {
            clearTimeout(this.turnOffTimer);
            this.turnOffTimer = null;
        }
        this.log.debug(`[State] 清理關燈計時器 (clearTimersAndStates)。`);
    }

    initialiseGPIO() {
        try {
            if (typeof this.relayPinNum === 'undefined') {
                 this.log.warn('[GPIO] Warning: Relay GPIO pin not configured. Light control via GPIO will be disabled.');
            } else {
                this.relayPin = new Gpio(this.relayPinNum, { mode: Gpio.OUTPUT });
            }

            if (typeof this.ldrPinNum === 'undefined') {
                 this.log.warn('[GPIO] Warning: LDR GPIO pin not configured. Darkness detection will assume "dark".');
            } else {
                this.ldrPin = new Gpio(this.ldrPinNum, { mode: Gpio.INPUT });
            }
            
            if (this.usePIR) {
                if (typeof this.pirPinNum === 'undefined') {
                    this.log.warn('[GPIO] Warning: PIR is enabled but GPIO pin not configured. PIR functionality will be disabled.');
                    this.usePIR = false; // 禁用PIR如果pin未定義
                } else {
                    this.pirPin = new Gpio(this.pirPinNum, { mode: Gpio.INPUT, alert: true });
                    this.pirPin.on('alert', (level, tick) => {
                        if (level === 1 ) { // PIR 偵測到移動 (高電平)
                            this.log.info(`[PIR Event] PIR triggered (level ${level}).`);
                            this.handlePirActivation();
                        } else { // PIR 恢復到低電平 (有些PIR會這樣)
                             this.log.debug(`[PIR Event] PIR signal low (level ${level}).`);
                        }
                    });
                    this.log.info('[GPIO] PIR已啟用並監聽中斷。');
                }
            } else {
                this.log.info('[GPIO] PIR is disabled by configuration.');
            }

            this.log.info(`[GPIO] 就緒 → Relay:${this.relayPinNum || 'N/A'}, LDR:${this.ldrPinNum || 'N/A (Assume Dark)'}` + (this.usePIR ? `, PIR:${this.pirPinNum || 'N/A (Disabled)'}` : ', PIR: Disabled'));
            this.setLightState(false, "[Init]"); 
            this.startAutomationLoop(); 
        } catch (err) {
            this.log.error(`[GPIO] 初始化失敗：${err.message}`);
            this.log.error(err.stack);
        }
    }
    
    setLightState(on, reason = "[Unknown]") {
        if (this.lightOn === on && reason !== "[Init]") {
            this.log.debug(`[Light] 狀態未變 (${on ? '開' : '關'}), ${reason}，無需操作。`);
            return;
        }
        if (!this.relayPin && reason !== "[Init]") { 
            this.log.warn(`[GPIO] Relay pin not initialized or configured. Cannot set light state for reason: ${reason}.`);
            if (reason === "[Init]") this.lightOn = on; 
            return;
        }
        try {
            const value = this.invertRelay ? (on ? 0 : 1) : (on ? 1 : 0);
            if (this.relayPin) this.relayPin.digitalWrite(value); 
            else if (reason !== "[Init]") this.log.warn(`[GPIO] Relay pin object not available during setLightState for reason: ${reason}`);

            this.lightOn = on;
            this.log.info(`[Light] 燈狀態設置為 -> ${this.lightOn ? '開' : '關'} (原因: ${reason})`);
            if (this.lightService) { 
                this.lightService.updateCharacteristic(Characteristic.On, this.lightOn);
            }
        } catch (err) {
            this.log.error(`[GPIO] 燈狀態 (${on}) 寫入失敗：${err.message}`);
            this.log.error(err.stack);
        }
    }
    
    handlePirActivation() { 
        if (!this.automationEnabled || this.manualOverrideActive || !this.usePIR) { // 增加 !this.usePIR 判斷
            this.log.debug(`[PIR Logic] Activation attempt ignored. Automation: ${this.automationEnabled}, Manual Override: ${this.manualOverrideActive}, UsePIR: ${this.usePIR}`);
            return;
        }
        this.log.info(`[PIR Event] PIR Detected motion (handlePirActivation).`);
        this.pirLastActivationTimestamp = Date.now();
        this.log.info(`[PIR Logic] Updated pirLastActivationTimestamp: ${this.pirLastActivationTimestamp}`);
        this.evaluateLightState(); 
    }

    startAutomationLoop() { 
        if (this.loopTimer) clearInterval(this.loopTimer); 
        this.loopTimer = setInterval(() => {
            if (!this.automationEnabled || this.manualOverrideActive) {
                return;
            }
            // CSI 數據超時邏輯
            if (this.csiPresenceDetected && (Date.now() - this.lastCsiDataTime > this.csiDataTimeoutMs)) {
                this.log.warn(`[Loop] CSI 數據超時 (${this.csiDataTimeoutMs / 1000}秒無新數據)，將CSI視為無人。`);
                if (this.csiPresenceDetected) {  // 只有在狀態真的改變時才打印
                    this.log.info(`[CSI Model] 預測狀態改變 (數據超時): 之前有人 -> 現在無人`);
                }
                this.csiPresenceDetected = false; 
                this.evaluateLightState();
            }
            // 定期評估燈光狀態，即使沒有CSI或PIR事件，以處理LDR變化或PIR超時
             else { 
                this.evaluateLightState();
            }
        }, this.scanIntervalMs); // 使用 scanIntervalMs 作為定期評估的間隔
        this.log.info(`[Automation] 自動化主循環 (CSI超時及定期狀態評估) 已啟動，間隔: ${this.scanIntervalMs}ms。`);
    }
    
    evaluateLightState() {
        if (!this.automationEnabled || this.manualOverrideActive) {
            this.log.debug('[Logic Eval] Skipped: Automation disabled or manual override active.');
            return;
        }

        const pirRecentlyTriggered = this.usePIR ? (Date.now() - this.pirLastActivationTimestamp < this.pirActiveWindowMs) : false; 
        const isCurrentlyDark = this.isDark();
        const csiSaysPresence = this.csiPresenceDetected; 

        this.log.info(`[Logic Eval] State | Light: ${this.lightOn}, CSI: ${csiSaysPresence}, PIR_recent: ${pirRecentlyTriggered}, Dark: ${isCurrentlyDark}, TimerActive: ${!!this.turnOffTimer}`);

        // 決策邏輯開始
        if (!this.lightOn) { // 如果燈是關的
            let turnOn = false;
            let reason = "";
            if (isCurrentlyDark) { // 只有天暗才考慮開燈
                if (this.usePIR) {
                    if (pirRecentlyTriggered && csiSaysPresence) {
                        turnOn = true;
                        reason = "[Auto] PIR + CSI Presence + Dark";
                    } else if (pirRecentlyTriggered && !csiSaysPresence) {
                        this.log.info('[Logic Decision] Light OFF. PIR triggered, but CSI says NO presence. Light remains OFF (CSI優先或需要CSI確認).');
                        // 根據您的描述 "一開始PIR要先偵測到人 然後CSI也要偵測到人"，所以PIR響應但CSI沒響應，則不開燈
                    } else if (!pirRecentlyTriggered && csiSaysPresence) {
                        // 根據您的描述 "一開始PIR要先偵測到人"，如果PIR未觸發，即使CSI說有人，初始也不開燈
                        this.log.info('[Logic Decision] Light OFF. CSI detected, but PIR NOT recently triggered. Light remains OFF (PIR has priority for initial ON).');
                    }
                } else { // 不使用PIR
                    if (csiSaysPresence) {
                        turnOn = true;
                        reason = "[Auto] CSI Presence (No PIR) + Dark";
                    }
                }
            } else { // 天亮了，不開燈
                 this.log.debug('[Logic Decision] Light OFF. It is light. No action.');
            }

            if (turnOn) {
                this.log.info(`[Logic Decision] Conditions to TURN ON met. Reason: ${reason}`);
                this.setLightState(true, reason);
                if (this.turnOffTimer) { // 如果因為某些原因有關燈計時器在跑，現在要開燈了就清除它
                    clearTimeout(this.turnOffTimer);
                    this.turnOffTimer = null;
                    this.log.debug('[Logic Decision] Turn ON: Cleared existing turnOffTimer.');
                }
            }
        } else { // 如果燈是亮的 (this.lightOn is true)
            if (csiSaysPresence) { 
                // CSI 說有人，燈應該保持亮，除非天變亮了
                if (!isCurrentlyDark) { 
                    this.log.info('[Logic Decision] Light ON, CSI presence, but it became LIGHT. Turning light OFF.');
                    this.setLightState(false, "[Auto] Became Light with Presence");
                    if (this.turnOffTimer) { clearTimeout(this.turnOffTimer); this.turnOffTimer = null; }
                } else { // 天仍然暗，CSI說有人，燈保持亮
                    this.log.info('[Logic Decision] Light ON, CSI presence, Dark. Light remains ON.');
                    if (this.turnOffTimer) { // 如果之前因為短暫無人而啟動了計時器，現在有人了就取消
                        this.log.info('[Logic Decision] Light ON, CSI presence confirmed. Cancelling turn OFF timer.');
                        clearTimeout(this.turnOffTimer);
                        this.turnOffTimer = null;
                    }
                }
            } else { // CSI 偵測到無人 (csiSaysPresence is false)
                // PIR的補救機制/重新倒數邏輯
                if (this.usePIR && pirRecentlyTriggered) {
                    this.log.info('[Logic Decision] Light ON, CSI NO presence, but PIR recently triggered. Light remains ON, cancelling turn OFF timer if any.');
                     if (this.turnOffTimer) { 
                        clearTimeout(this.turnOffTimer);
                        this.turnOffTimer = null;
                    }
                } else { // CSI無人，且PIR也未在活躍窗口期內觸發 (或者不使用PIR)
                    if (!this.turnOffTimer) { // 如果還沒有啟動關燈計時器
                        this.log.info(`[Logic Decision] Light ON, CSI NO presence (PIR not recent or not used). Starting ${this.turnOffDelayMs / 1000}s turn OFF timer.`);
                        this.turnOffTimer = setTimeout(() => {
                            this.log.info('[Logic Decision] Turn OFF timer expired. Re-checking final conditions before turning light OFF.');
                            const currentCsiSaysPresenceAfterDelay = this.csiPresenceDetected; 
                            const currentPirRecentlyTriggeredAfterDelay = this.usePIR ? (Date.now() - this.pirLastActivationTimestamp < this.pirActiveWindowMs) : false;
                            
                            if (!currentCsiSaysPresenceAfterDelay && (!this.usePIR || !currentPirRecentlyTriggeredAfterDelay)) { 
                                this.log.info('[Logic Decision] Confirmed NO presence (CSI and PIR) after delay. Turning light OFF.');
                                this.setLightState(false, "[Auto] CSI No Presence Delay");
                            } else {
                                this.log.info('[Logic Decision] Turn OFF timer expired, but presence (CSI or recent PIR) detected again. Light remains ON.');
                                // 這裡不需要再次調用 evaluateLightState，因為PIR或CSI的下一個狀態變化會觸發它
                            }
                            this.turnOffTimer = null; 
                        }, this.turnOffDelayMs);
                    } else {
                        this.log.debug('[Logic Decision] Light ON, CSI NO presence. Turn OFF timer already active.');
                    }
                }
            }
        }
    }

    isDark() { 
        if (typeof this.ldrPinNum === 'undefined' || this.ldrPinNum === null) {
            this.log.debug("[Logic Eval] LDR pin not configured, assuming it's dark for automation logic.");
            return true; 
        }
        return this._safeRead(this.ldrPin) === 1; // 假設1代表暗
    }
    _safeRead(pin) {
        try { 
            if (!pin) { // 再次檢查 pin 對象是否存在
                this.log.warn(`[GPIO] Attempted to read uninitialized pin object (likely LDR or PIR if not configured).`);
                return 1; 
            }
            return pin.digitalRead() ?? 1; 
        }
        catch (readError) {
            this.log.warn(`[GPIO] Failed to read pin: ${readError.message}`);
            return 1; 
        }
    }

    unloadCallback() { 
        this.log.info('執行 unloadCallback，清理資源...');
        clearInterval(this.loopTimer);
        this.loopTimer = null;
        if (this.turnOffTimer) clearTimeout(this.turnOffTimer);
        this.turnOffTimer = null;
        if (this.manualOverrideTimer) clearTimeout(this.manualOverrideTimer);
        this.manualOverrideTimer = null;
        
        if (this.mqttClient) {
            this.mqttClient.end(true, () => {
                this.log.info('[MQTT] 連線已關閉。');
            });
            this.mqttClient = null;
        }

        if (this.pythonProcess) {
            this.log.info('[PythonService] Killing Python process on unload.');
            this.pythonProcess.kill('SIGTERM'); 
            this.pythonProcess = null;
        }

        try {
            if (this.lightOn && this.relayPin && typeof this.relayPin.digitalWrite === 'function') {
                 this.relayPin.digitalWrite(this.invertRelay ? 1 : 0); 
            }
            this.relayPin?.unexport?.();
            this.ldrPin?.unexport?.();
            this.pirPin?.unexport?.();
            this.log.info('[GPIO] GPIO已釋放。');
        } catch (err) {
            this.log.warn(`[GPIO] 釋放失敗：${err.message}`);
        }
    }

    getServices() {
        return [this.informationService, this.lightService, this.automationSwitchService];
    }
}
