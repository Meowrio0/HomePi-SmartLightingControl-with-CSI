#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_csi_data_collector_v3_rate_control.py
接收從 ESP32 通過 UDP 發送的 CSI 數據 (JSON格式)，
並按指定的目標頻率有選擇性地保存 csi_raw, rssi, mcs 以及手動設定的 class 和 scenario 到 CSV 文件。
"""

import socket
import time
import csv
import pathlib
import json 
from datetime import datetime
import traceback 
import sys
import numpy as np 

# ---------- ★★ UDP 監聽設定 ★★ ----------
UDP_IP   = "0.0.0.0"
UDP_PORT = 5555       
BUFFER_SIZE_UDP = 2048 
# ----------------------------------------------------

# ---------- ★★ 採集設定 (每次採集前請修改這些) ★★ ----------
OCCUPANCY_DEFAULT = 0  # 0 = 無人, 1 = 有人 
SCENARIO_DEFAULT  = 'no_person_udp_16hz-2' # 【建議每次採集都修改此名稱】
CAPTURE_SECONDS = 7200      # 收集時長 (秒)。
WAIT_SECONDS    = 120        

# *** 新增：目標保存頻率設定 ***
TARGET_SAVE_HZ = 30.0  # 您期望保存到 CSV 文件中的數據頻率 (例如 100Hz 或 50Hz)
# ------------------------------------------------------------

# 計算保存間隔
SAVE_INTERVAL_SEC = 1.0 / TARGET_SAVE_HZ if TARGET_SAVE_HZ > 0 else 0

# 輸出文件設定
desktop = pathlib.Path.home() / 'Desktop'
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_scenario_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in SCENARIO_DEFAULT).rstrip()
output_filename = f"{safe_scenario_name}_{timestamp_str}_target{int(TARGET_SAVE_HZ)}hz.csv" # 文件名中加入目標頻率
outfile = desktop / output_filename

HEADER = ["data", "class", "scenario", "rssi", "mcs", "pc_timestamp"]

def main():
    print(f"[INFO] {WAIT_SECONDS} 秒後開始收集 UDP 數據...")
    print(f"[INFO] 設定狀態: {'有人' if OCCUPANCY_DEFAULT == 1 else '無人'}, 場景: {SCENARIO_DEFAULT}")
    print(f"[INFO] 預計收集時長: {CAPTURE_SECONDS} 秒")
    print(f"[INFO] 目標保存頻率: {TARGET_SAVE_HZ:.1f} Hz (每 ~{SAVE_INTERVAL_SEC:.4f} 秒保存一個包)")
    print(f"[INFO] 數據將儲存到: {outfile}")
    time.sleep(WAIT_SECONDS)

    sock = None
    csv_file = None
    csv_writer = None
    packets_received_total = 0
    packets_saved_total = 0
    last_saved_time = 0.0 # 初始化上次保存數據包的時間
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        # 設置一個較短的 socket 超時，以便 KeyboardInterrupt 能更快響應 (如果需要)
        sock.settimeout(1.0) # 1秒超時
        print(f"[INFO] UDP Socket 已綁定到 {UDP_IP}:{UDP_PORT}。開始接收數據... (按 Ctrl+C 可提前結束)")
        
        start_ts = time.time()

        csv_file = open(outfile, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(HEADER) 

        while time.time() - start_ts < CAPTURE_SECONDS:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE_UDP) 
                reception_timestamp = time.time() # 精確的接收時間
                packets_received_total += 1
                message_str = data.decode('utf-8', errors='ignore').strip()
                
                # 僅當達到保存時間間隔時才處理和保存
                if SAVE_INTERVAL_SEC == 0 or (reception_timestamp - last_saved_time >= SAVE_INTERVAL_SEC):
                    try:
                        data_json = json.loads(message_str)
                        csi_raw_value = data_json.get('csi_raw', None) 
                        rssi_value = data_json.get('rssi', np.nan)
                        mcs_value = data_json.get('mcs', np.nan)
                        
                        if csi_raw_value is not None and isinstance(csi_raw_value, str):
                            csv_writer.writerow([
                                csi_raw_value, OCCUPANCY_DEFAULT, SCENARIO_DEFAULT,
                                rssi_value, mcs_value, reception_timestamp     
                            ])
                            packets_saved_total += 1
                            last_saved_time = reception_timestamp # 更新上次保存時間
                            
                            if packets_saved_total % 100 == 0: # 每保存100個包打印一次計數
                                 current_elapsed_total = time.time() - start_ts
                                 current_avg_save_rate = packets_saved_total / current_elapsed_total if current_elapsed_total > 0 else 0
                                 print(f"[INFO] 已保存 {packets_saved_total} 個 CSI 數據包... (當前平均保存速率約: {current_avg_save_rate:.2f} Hz)")
                        # else: # 可選：如果 csi_raw 格式不對
                        #     if sys.stdout.isatty(): print(f"[WARN] 'csi_raw' 格式不正確或缺失 (來自 {addr}): {str(csi_raw_value)[:50]}...")
                    except json.JSONDecodeError:
                        if sys.stdout.isatty(): print(f"[WARN] 無法解析 UDP 數據為 JSON (來自 {addr}): {message_str[:100]}...")
                    except Exception as e_parse:
                        if sys.stdout.isatty(): print(f"[ERROR] 解析或寫入時出錯: {e_parse}")
                # else: # 如果未達到保存間隔，則丟棄此包 (不打印，避免刷屏)
                    # pass 

            except socket.timeout: 
                # print("UDP recvfrom timeout (正常，繼續等待)") # 這個打印太頻繁
                continue # 繼續等待下一個數據包
            except Exception as e_sock:
                if sys.stdout.isatty(): print(f"[ERROR] UDP socket 接收錯誤: {e_sock}")
                time.sleep(0.1)

        actual_duration = time.time() - start_ts
        print(f"\n[INFO] 收集完成！")
        print(f"[INFO] 實際收集時長: {actual_duration:.2f} 秒")
        print(f"[INFO] 共接收到 {packets_received_total} 個 UDP 數據包")
        print(f"[INFO] 共選擇並寫入 {packets_saved_total} 個 CSI 數據包到 CSV")
        if actual_duration > 0 and packets_saved_total > 0 :
            avg_save_rate = packets_saved_total / actual_duration
            print(f"[INFO] 平均有效 CSI 保存速率: {avg_save_rate:.2f} Hz (目標: {TARGET_SAVE_HZ:.1f} Hz)")
        else:
            print(f"[INFO] 未收集到有效 CSI 數據包。")
        print(f"[INFO] 檔案已儲存到: {outfile}")

    except KeyboardInterrupt:
        print("\n[INFO] 用戶手動停止收集。")
    except OSError as e:
        print(f"[ERROR] Socket 錯誤: {e}"); traceback.print_exc()
    except Exception as e_main:
        print(f"[ERROR] 發生未知錯誤: {e_main}"); traceback.print_exc()
    finally:
        if sock: sock.close(); print("[INFO] UDP Socket 已關閉。")
        if csv_file: csv_file.close(); print("[INFO] CSV 文件已關閉。")

if __name__ == '__main__':
    main()