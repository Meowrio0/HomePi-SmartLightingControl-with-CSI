#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mqtt_csi_hz_tester.py
連接到 MQTT Broker，訂閱指定的 CSI 數據主題，
並計算接收到的消息頻率。
"""

import paho.mqtt.client as mqtt
import time
import sys
import json # 可選，用於打印時解析payload

# ANSI Escape Codes for colors (與您之前的腳本一致)
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

# ---------- ★★ MQTT Broker 設定 (請根據您的實際情況修改) ★★ ----------
MQTT_BROKER_IP = "192.168.1.11"  # 【重要】替換為您的 MQTT Broker (例如樹莓派) 的實際 IP 地址
MQTT_BROKER_PORT = 1883          # 常用的 MQTT 端口
MQTT_TOPIC = "homebridge/csi/raw_data" # 【重要】必須與您 ESP32 MQTT 固件中發布的主題一致
# 如果您的 MQTT Broker 需要用戶名和密碼：
# MQTT_USERNAME = "your_username"
# MQTT_PASSWORD = "your_password"
# ---------------------------------------------------------------------

IS_INTERACTIVE_RUN = sys.stdin.isatty() and __name__ == '__main__'

# --- 全局變量用於速率計算 ---
packet_count = 0
calculation_start_time = 0
last_sample_print_time = 0 # 用於控制樣本消息的打印頻率

# MQTT 連接成功時的回調
def on_connect(client, userdata, flags, reason_code, properties): # Paho MQTT v1.6+ and v2.x
# def on_connect(client, userdata, flags, rc): # Uncomment for older Paho MQTT v1.5.x and below
    global calculation_start_time, packet_count, last_sample_print_time
    if reason_code == 0: # For Paho MQTT v1.6+
    # if rc == 0: # For older Paho MQTT v1.5.x and below
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKGREEN}成功連接到 MQTT Broker: {MQTT_BROKER_IP}{TermColors.ENDC}", file=sys.stderr)
        client.subscribe(MQTT_TOPIC)
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}已訂閱主題: {MQTT_TOPIC}{TermColors.ENDC}", file=sys.stderr)
        # 重置計數器和時間
        packet_count = 0
        calculation_start_time = time.time()
        last_sample_print_time = time.time()
    else:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}連接 MQTT Broker 失敗, reason code {reason_code}{TermColors.ENDC}", file=sys.stderr)
        # if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}連接 MQTT Broker 失敗, return code {rc}{TermColors.ENDC}", file=sys.stderr) # For older Paho

# 收到消息時的回調
def on_message(client, userdata, msg):
    global packet_count, last_sample_print_time
    packet_count += 1
    current_time = time.time()

    # 可選：每隔一段時間打印一小部分接收到的數據，以確認數據流正常
    if IS_INTERACTIVE_RUN and (current_time - last_sample_print_time >= 1.0): # 每秒打印一次樣本信息
        try:
            payload_str = msg.payload.decode('utf-8', errors='ignore')
            # 嘗試解析JSON以更美觀地打印，如果失敗則打印原始字符串片段
            try:
                data_json = json.loads(payload_str)
                csi_raw_snippet = data_json.get('csi_raw', 'N/A')[:30]
                rssi_val = data_json.get('rssi', 'N/A')
                mcs_val = data_json.get('mcs', 'N/A')
                print(f"  {TermColors.OKBLUE}收到MQTT (主題 {msg.topic}): csi_raw[:30]={csi_raw_snippet}..., rssi={rssi_val}, mcs={mcs_val}{TermColors.ENDC}", file=sys.stderr)
            except json.JSONDecodeError:
                print(f"  {TermColors.OKBLUE}收到MQTT (主題 {msg.topic}, 前30字元): {payload_str[:30]}...{TermColors.ENDC}", file=sys.stderr)
        except Exception as e_decode:
            print(f"  {TermColors.WARNING}解碼或打印某個MQTT消息時出錯: {e_decode}{TermColors.ENDC}", file=sys.stderr)
        last_sample_print_time = current_time

def run_mqtt_hz_test():
    global packet_count, calculation_start_time

    # 注意 Paho MQTT 版本。對於 v1.6+ 或 v2.x，使用 CallbackAPIVersion.VERSION2
    # 如果您的 paho-mqtt 版本較舊 (例如 1.5.x)，請使用 client = mqtt.Client()
    # 並相應調整 on_connect 的參數 (移除 properties, reason_code -> rc)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) 
    client.on_connect = on_connect
    client.on_message = on_message

    # 如果您的 Broker 需要用戶名密碼
    # client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    try:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}嘗試連接到 MQTT Broker {MQTT_BROKER_IP}:{MQTT_BROKER_PORT}...{TermColors.ENDC}", file=sys.stderr)
        client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)
    except Exception as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: 連接 MQTT Broker 失敗: {e}{TermColors.ENDC}", file=sys.stderr)
        else: print(f"FatalError: Cannot connect to MQTT Broker: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    client.loop_start() # 在後台線程中運行網絡循環

    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.WARNING}請確保您的ESP32 (MQTT版本固件) 正在運行，並發布數據到主題 '{MQTT_TOPIC}' 至 Broker '{MQTT_BROKER_IP}'.{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.WARNING}您可以嘗試在ESP32固件中修改 PING_FREQUENCY_HZ 的值來觀察速率變化。{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.OKBLUE}按 Ctrl+C 停止測試。{TermColors.ENDC}", file=sys.stderr)

    display_interval_seconds = 5.0  # 每5秒計算並打印一次平均速率

    try:
        while True:
            time.sleep(display_interval_seconds) # 主線程等待 display_interval_seconds
            
            current_time = time.time()
            elapsed_time = current_time - calculation_start_time
            
            if elapsed_time > 0: # 避免除以零 (雖然 time.sleep 保證了 elapsed_time > 0)
                rate = packet_count / elapsed_time
                if IS_INTERACTIVE_RUN:
                    print(f"{TermColors.OKGREEN}>>> 在過去 {elapsed_time:.2f} 秒內，平均接收速率: {TermColors.BOLD}{rate:.2f} Hz{TermColors.ENDC} ({packet_count} 個MQTT消息)", file=sys.stderr)
                else: # 非交互模式，直接打印速率到 stdout
                    print(f"Rate: {rate:.2f} Hz")
                
                # 重置計數器和開始時間
                packet_count = 0
                calculation_start_time = current_time
            elif packet_count > 0 and IS_INTERACTIVE_RUN: # 如果時間間隔太短但收到了包
                 print(f"{TermColors.OKCYAN}在極短時間內收到 {packet_count} 個包，等待下個統計週期。{TermColors.ENDC}", file=sys.stderr)


    except KeyboardInterrupt:
        if IS_INTERACTIVE_RUN: print(f"\n{TermColors.FAIL}測試被用戶終止。{TermColors.ENDC}", file=sys.stderr)
    finally:
        client.loop_stop() # 停止後台網絡線程
        client.disconnect()
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}MQTT客戶端已斷開連接。{TermColors.ENDC}", file=sys.stderr)

if __name__ == '__main__':
    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.HEADER}=== ESP32 CSI MQTT 數據發送頻率測試工具 ==={TermColors.ENDC}", file=sys.stderr)
    try:
        run_mqtt_hz_test()
    except SystemExit:
        pass
    except Exception as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: HZ測試腳本主函數未處理異常: {e}{TermColors.ENDC}", file=sys.stderr)
        else: print(f"FatalError: Unhandled error in HZ test script: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)