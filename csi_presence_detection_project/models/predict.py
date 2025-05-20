import pandas as pd
import numpy as np
import pickle
# import socket #不再直接使用UDP
import json
import time
import os
import sys
import traceback
from collections import deque # 用於管理數據緩衝區

import tensorflow as tf # 用於加載TFLite模型

# ANSI Escape Codes for colors
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

IS_INTERACTIVE_RUN = sys.stdin.isatty() and __name__ == '__main__' # 判斷是否交互運行以控制打印

# --- 0. Utility Functions (與 terminal.py / model.py 一致，用於CSI解析) ---
def parse_csi_raw_to_int_array(csi_string, expected_length=None):
    if pd.isna(csi_string):
        return np.full(expected_length, np.nan) if expected_length is not None else []
    try:
        s = str(csi_string).strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        if not s:
            return np.full(expected_length, np.nan) if expected_length is not None else []
        csi_values = [int(x) for x in s.split(',')]
        if expected_length is not None and len(csi_values) != expected_length:
            return np.full(expected_length, np.nan)
        return np.array(csi_values)
    except Exception:
        return np.full(expected_length, np.nan) if expected_length is not None else []

NUM_META_ELEMENTS = 0 

def calculate_amplitudes_from_iq_pairs(iq_array, meta_elements=0, current_expected_amplitude_len=64): # 加入期望長度參數
    if not isinstance(iq_array, np.ndarray) or pd.isna(iq_array).all() or len(iq_array) <= meta_elements:
        return np.full(current_expected_amplitude_len, np.nan)
    actual_iq_data = iq_array[meta_elements:]
    if len(actual_iq_data) == 0 or len(actual_iq_data) % 2 != 0:
        return np.full(current_expected_amplitude_len, np.nan)
    amplitudes = []
    for i in range(0, len(actual_iq_data), 2):
        I = actual_iq_data[i]
        Q = actual_iq_data[i+1]
        if pd.isna(I) or pd.isna(Q):
            amplitudes.append(np.nan)
        else:
            amplitude = np.sqrt(float(I)**2 + float(Q)**2)
            amplitudes.append(amplitude)
    
    if len(amplitudes) != current_expected_amplitude_len: 
        return np.full(current_expected_amplitude_len, np.nan)
    return np.array(amplitudes)

# 新增：從 terminal.py 引入 select_subcarriers_from_esp32_data
def select_subcarriers_from_esp32_data(amplitudes_raw_esp32):
    global N_SUBCARRIERS_FOR_MODEL, RAW_AMPLITUDE_LEN_FROM_ESP32, WALLHACK_LLTF_SUBCARRIER_INDICES
    if not isinstance(amplitudes_raw_esp32, np.ndarray):
        return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
    if len(amplitudes_raw_esp32) == RAW_AMPLITUDE_LEN_FROM_ESP32: # RAW_AMPLITUDE_LEN_FROM_ESP32 應為64
        try:
            if not WALLHACK_LLTF_SUBCARRIER_INDICES: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
            if max(WALLHACK_LLTF_SUBCARRIER_INDICES) < RAW_AMPLITUDE_LEN_FROM_ESP32:
                selected_amps = amplitudes_raw_esp32[WALLHACK_LLTF_SUBCARRIER_INDICES]
                if len(selected_amps) == N_SUBCARRIERS_FOR_MODEL: return selected_amps
                else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
            else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
        except IndexError: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
    else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)


# --- 全局常量 (部分將由元數據覆蓋) ---
EXPECTED_INT_ARRAY_LEN = 128 
RAW_AMPLITUDE_LEN_FROM_ESP32 = (EXPECTED_INT_ARRAY_LEN - NUM_META_ELEMENTS) // 2 # 64
WALLHACK_LLTF_SUBCARRIER_INDICES = list(range(6, 32)) + list(range(33, 59)) # 與 model.py 一致
N_SUBCARRIERS_FOR_MODEL = 52 # 與 model.py 一致

SAMPLING_RATE_HZ = 16.0 # 將由元數據覆蓋
WINDOW_DURATION_SEC = 1.5 # 僅用於計算初始WINDOW_SIZE
_INITIAL_WINDOW_SIZE = int(SAMPLING_RATE_HZ * WINDOW_DURATION_SEC) # 例如 16 * 1.5 = 24
WINDOW_SIZE = _INITIAL_WINDOW_SIZE # 將由元數據的 'sequence_length' 覆蓋

# PACKET_LEVEL_BASE_FEATURES_FOR_STATS 將由元數據的 'feature_names' 推斷
PACKET_LEVEL_BASE_FEATURES_FOR_STATS = ['csi_amp_std'] # 默認，會被元數據覆蓋邏輯調整
LSTM_INPUT_FEATURES = [] # 將從元數據加載
N_LSTM_FEATURES = 0 # 將從元數據加載
RECONSTRUCTION_ERROR_THRESHOLD = 0.5 # 將從元數據加載

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
DATA_BUFFER_FILE = os.path.join(SCRIPT_DIR, 'csi_data_buffer_lstm_ae.json') # 新的緩存文件名

# --- 模型文件路徑 ---
MODEL_PARENT_DIR = "/HomePi-SmartLighting/homebridge-smart-light-csi/python/" # 【請確認此路徑】
# 【重要】指向您最新訓練的 LSTM Autoencoder 的元數據和模型文件
ACTUAL_METADATA_FILENAME = 'csi_lstm_ae_S16hz_L24_NO_DIFF_metadata.pkl' 
ACTUAL_TFLITE_FILENAME_DEFAULT = 'csi_lstm_ae_S16hz_L24_NO_DIFF_float32.tflite'

METADATA_FILE_PKL = os.path.join(MODEL_PARENT_DIR, ACTUAL_METADATA_FILENAME)

# --- 加載模型和元數據 ---
loaded_interpreter = None
scaler = None
model_metadata = {}

try:
    if not os.path.exists(METADATA_FILE_PKL):
        # 在非交互模式下，如果元數據不存在，是很嚴重的問題
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: 元數據文件 {METADATA_FILE_PKL} 未找到。{TermColors.ENDC}", file=sys.stderr)
        else: sys.stderr.write(f"FatalError: Metadata file {METADATA_FILE_PKL} not found.\n")
        sys.exit(1) # 直接退出
        
    with open(METADATA_FILE_PKL, 'rb') as f:
        model_metadata = pickle.load(f)
    if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE}從 {METADATA_FILE_PKL} 加載了元數據。{TermColors.ENDC}", file=sys.stderr)

    WINDOW_SIZE = int(model_metadata.get('sequence_length', _INITIAL_WINDOW_SIZE))
    LSTM_INPUT_FEATURES = list(model_metadata.get('feature_names', ['csi_amp_std'])) # 默認為單特徵
    N_LSTM_FEATURES = int(model_metadata.get('n_features', len(LSTM_INPUT_FEATURES)))
    RECONSTRUCTION_ERROR_THRESHOLD = float(model_metadata.get('reconstruction_error_threshold', 0.5))
    SAMPLING_RATE_HZ = float(model_metadata.get('sampling_rate_hz_used', SAMPLING_RATE_HZ))

    if all(not f.endswith("_diff") for f in LSTM_INPUT_FEATURES):
        PACKET_LEVEL_BASE_FEATURES_FOR_STATS = list(LSTM_INPUT_FEATURES)
    else:
        PACKET_LEVEL_BASE_FEATURES_FOR_STATS = list(set([f.replace("_diff","") for f in LSTM_INPUT_FEATURES]))
    
    if 'scaler_object' in model_metadata and model_metadata['scaler_object'] is not None:
        scaler = model_metadata['scaler_object']
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKGREEN}[DEBUG INIT] StandardScaler 從元數據加載成功。{TermColors.ENDC}", file=sys.stderr)
    else:
        # 如果模型訓練時用了標準化（由元數據中的 'data_scaled' 標記判斷，若無則默認為True）
        # 但元數據中卻沒有 scaler_object，這是一個嚴重問題。
        data_was_scaled_in_training = model_metadata.get('data_scaled', True) 
        if data_was_scaled_in_training:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[CRITICAL WARNING] 元數據中未找到 'scaler_object'，但模型可能基於標準化數據訓練！推斷將不準確。{TermColors.ENDC}", file=sys.stderr)
            else: sys.stderr.write("[CRITICAL WARNING] Scaler missing in metadata, but model likely expects scaled data!\n")
            # 在非交互模式下，甚至可以考慮直接退出
        else: # 元數據明確指出 data_scaled: False
            if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}[DEBUG INIT] 元數據指示訓練數據未經標準化。實時推斷亦不使用標準化。{TermColors.ENDC}", file=sys.stderr)


    tflite_filename_from_meta = model_metadata.get('tflite_model_colab_path')
    actual_tflite_filename_to_load = os.path.basename(tflite_filename_from_meta) if tflite_filename_from_meta else ACTUAL_TFLITE_FILENAME_DEFAULT
    final_tflite_path_to_load = os.path.join(MODEL_PARENT_DIR, actual_tflite_filename_to_load)

    if not os.path.exists(final_tflite_path_to_load):
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: TFLite模型文件 {final_tflite_path_to_load} 未找到。{TermColors.ENDC}", file=sys.stderr)
        else: sys.stderr.write(f"FatalError: TFLite model {final_tflite_path_to_load} not found.\n")
        sys.exit(1)
        
    loaded_interpreter = tf.lite.Interpreter(model_path=final_tflite_path_to_load)
    loaded_interpreter.allocate_tensors()
    if IS_INTERACTIVE_RUN: print(f"{TermColors.OKGREEN}TFLite Autoencoder模型 {final_tflite_path_to_load} 加載成功。{TermColors.ENDC}", file=sys.stderr)

except Exception as e:
    if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: 初始化或加載模型/元數據過程中發生嚴重錯誤: {e}{TermColors.ENDC}", file=sys.stderr)
    else: sys.stderr.write(f"FatalError loading model/metadata: {e}\n")
    traceback.print_exc(file=sys.stderr if IS_INTERACTIVE_RUN else sys.stdout) # 非交互時打印到stdout
    sys.exit(1)


# --- 核心處理函數 (類似 terminal.py) ---
def process_csi_packet_to_features(csi_raw_str_from_json):
    # 此函數處理單個包，提取基礎特徵
    # PACKET_LEVEL_BASE_FEATURES_FOR_STATS 應已根據元數據設定
    packet_features = {}
    
    csi_parsed_int_list = parse_csi_raw_to_int_array(csi_raw_str_from_json, EXPECTED_INT_ARRAY_LEN)
    csi_amplitudes_raw_esp32 = calculate_amplitudes_from_iq_pairs(csi_parsed_int_list, meta_elements=NUM_META_ELEMENTS, current_expected_amplitude_len=RAW_AMPLITUDE_LEN_FROM_ESP32)
    
    if isinstance(csi_amplitudes_raw_esp32, np.ndarray) and not np.isnan(csi_amplitudes_raw_esp32).all():
        selected_amplitudes = select_subcarriers_from_esp32_data(csi_amplitudes_raw_esp32)
        if isinstance(selected_amplitudes, np.ndarray) and len(selected_amplitudes) == N_SUBCARRIERS_FOR_MODEL and not np.isnan(selected_amplitudes).any():
            for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS:
                if base_feat_name == 'csi_amp_std':
                    packet_features['csi_amp_std'] = np.std(selected_amplitudes)
                # 如果將來 PACKET_LEVEL_BASE_FEATURES_FOR_STATS 包含其他特徵，在此處添加計算邏輯
                else:
                    packet_features[base_feat_name] = np.nan 
        else: # selected_amplitudes 無效
            for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: packet_features[base_feat_name] = np.nan
    else: # csi_amplitudes_raw_esp32 無效
        for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: packet_features[base_feat_name] = np.nan
            
    packet_features['timestamp'] = time.time()
    return packet_features

def prepare_feature_sequence_for_model(window_packet_features_list):
    # 此函數從包特徵列表準備模型輸入序列，並進行標準化
    # LSTM_INPUT_FEATURES 和 N_LSTM_FEATURES 應已從元數據加載
    global WINDOW_SIZE, LSTM_INPUT_FEATURES, N_LSTM_FEATURES, scaler # scaler 是全局的

    if len(window_packet_features_list) != WINDOW_SIZE: return None
    df_window = pd.DataFrame(window_packet_features_list)
    if df_window.empty: return None

    # 根據 LSTM_INPUT_FEATURES 計算差分特徵 (如果需要)
    # PACKET_LEVEL_BASE_FEATURES_FOR_STATS 此時應該是 LSTM_INPUT_FEATURES 中非 _diff 的部分
    for base_feat in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: 
        diff_col_name = f'{base_feat}_diff'
        if diff_col_name in LSTM_INPUT_FEATURES and base_feat in df_window.columns:
            df_window[diff_col_name] = df_window[base_feat].ffill().bfill().diff().fillna(0)
    
    # 確保所有期望的列都存在，即使是剛計算的差分特徵
    for col in LSTM_INPUT_FEATURES:
        if col not in df_window.columns:
            df_window[col] = 0.0 # 如果某個期望的特徵列不存在（例如差分計算失敗），填充0

    try:
        sequence_data_raw = df_window[LSTM_INPUT_FEATURES].values # (WINDOW_SIZE, N_FEATURES)
    except KeyError:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}Prepare sequence: 特徵提取失敗，可能是列缺失。{TermColors.ENDC}", file=sys.stderr)
        return None

    if np.isnan(sequence_data_raw).any() or np.isinf(sequence_data_raw).any():
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}Prepare sequence: 原始序列包含NaN/Inf，嘗試填充0。{TermColors.ENDC}", file=sys.stderr)
        sequence_data_raw = np.nan_to_num(sequence_data_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(sequence_data_raw).any() or np.isinf(sequence_data_raw).any(): # 再次檢查
             if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}Prepare sequence: 填充後序列仍包含NaN/Inf。{TermColors.ENDC}", file=sys.stderr)
             return None


    if sequence_data_raw.shape != (WINDOW_SIZE, N_LSTM_FEATURES):
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}Prepare sequence: 原始序列形狀 {sequence_data_raw.shape} 與預期 ({WINDOW_SIZE}, {N_LSTM_FEATURES}) 不符。{TermColors.ENDC}", file=sys.stderr)
        return None

    # 標準化 (如果 scaler 存在且元數據指示需要標準化)
    sequence_data_scaled = np.copy(sequence_data_raw) # 默認不縮放或scaler不存在
    data_scaled_info = model_metadata.get('data_scaled', True) # 從元數據獲取是否標準化的信息

    if scaler is not None and data_scaled_info:
        try:
            # Scaler 期望 (n_samples, n_features)，我們的序列是 (WINDOW_SIZE, N_FEATURES)
            # reshape 不必要，因為 scaler.transform 可以直接處理 2D array
            sequence_data_scaled = scaler.transform(sequence_data_raw) 
        except Exception as e_scale:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}Prepare sequence: 標準化時出錯: {e_scale}。將使用原始序列。{TermColors.ENDC}", file=sys.stderr)
            sequence_data_scaled = sequence_data_raw # 出錯則退回使用原始數據
    elif not data_scaled_info and IS_INTERACTIVE_RUN :
        print(f"{TermColors.OKCYAN}Prepare sequence: 元數據指示不進行標準化，使用原始特徵序列。{TermColors.ENDC}", file=sys.stderr)


    return np.reshape(sequence_data_scaled, (1, WINDOW_SIZE, N_LSTM_FEATURES))


def main():
    # 不再使用 argparse，改為從 stdin 讀取 JSON 行
    # parser = argparse.ArgumentParser(description="Predict occupancy based on CSI data.")
    # parser.add_argument('--csi_raw', type=str, required=True, help='CSI raw data string, e.g., "[...]"')
    # args = parser.parse_args()
    # csi_raw_from_arg = args.csi_raw

    line = sys.stdin.readline()
    if not line:
        if IS_INTERACTIVE_RUN: print(" predict.py: No input from stdin. Exiting.", file=sys.stderr)
        print("buffering") # 表示等待數據
        sys.exit(0) # 正常退出，等待下次調用

    try:
        data_json_from_stdin = json.loads(line)
        csi_raw_from_stdin = data_json_from_stdin.get('csi_raw')
        # 可選：從 data_json_from_stdin 提取 rssi, mcs，但當前模型不使用它們
    except json.JSONDecodeError:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: 無法解析來自 stdin 的 JSON: {line.strip()[:100]}...{TermColors.ENDC}", file=sys.stderr)
        print("error") # 表示解析錯誤
        sys.exit(1)
    except KeyError:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: stdin 的 JSON 中缺少 'csi_raw' 鍵。{TermColors.ENDC}", file=sys.stderr)
        print("error") 
        sys.exit(1)

    if csi_raw_from_stdin is None:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: 從 stdin 接收到的 'csi_raw' 為空。{TermColors.ENDC}", file=sys.stderr)
        print("buffering")
        sys.exit(0)

    if IS_INTERACTIVE_RUN:
        # print(f"{TermColors.HEADER}--- 接收到新數據 (來自 stdin) ---{TermColors.ENDC}", file=sys.stderr) # 避免過多打印
        pass


    data_buffer_packet_features = []
    if os.path.exists(DATA_BUFFER_FILE):
        try:
            with open(DATA_BUFFER_FILE, 'r') as f:
                data_buffer_packet_features = json.load(f)
            if not isinstance(data_buffer_packet_features, list): data_buffer_packet_features = []
        except Exception:
            data_buffer_packet_features = []
    
    new_packet_feature_sample = process_csi_packet_to_features(csi_raw_from_stdin)
    data_buffer_packet_features.append(new_packet_feature_sample)

    if len(data_buffer_packet_features) > WINDOW_SIZE:
        data_buffer_packet_features = data_buffer_packet_features[-WINDOW_SIZE:]
    
    try:
        with open(DATA_BUFFER_FILE, 'w') as f: json.dump(data_buffer_packet_features, f)
    except Exception as e_write_buffer:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}警告: 保存數據緩存到 {DATA_BUFFER_FILE} 失敗: {e_write_buffer}{TermColors.ENDC}", file=sys.stderr)

    prediction_result_int = 0 # 默認為 "No Person"
    
    if len(data_buffer_packet_features) == WINDOW_SIZE:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}  包級別特徵緩存已滿 ({WINDOW_SIZE} 個)，準備使用 LSTM AE 預測...{TermColors.ENDC}", file=sys.stderr)
        
        feature_sequence_for_model = prepare_feature_sequence_for_model(list(data_buffer_packet_features))
        
        if feature_sequence_for_model is not None:
            try:
                input_details = loaded_interpreter.get_input_details()[0]
                output_details = loaded_interpreter.get_output_details()[0]
                
                input_data_final = feature_sequence_for_model
                if input_data_final.dtype != input_details['dtype']:
                    input_data_final = input_data_final.astype(input_details['dtype'])
                
                if input_data_final.shape != tuple(input_details['shape']):
                     if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}  TFLite 輸入形狀不匹配! 期望: {input_details['shape']}, 實際: {input_data_final.shape}{TermColors.ENDC}", file=sys.stderr)
                     print("error") # 輸出錯誤信號給 index.js
                     sys.exit(1)

                loaded_interpreter.set_tensor(input_details['index'], input_data_final)
                loaded_interpreter.invoke()
                reconstructed_sequence = loaded_interpreter.get_tensor(output_details['index'])
                
                reconstruction_error = np.mean(np.square(input_data_final - reconstructed_sequence))
                
                is_person_detected = reconstruction_error > RECONSTRUCTION_ERROR_THRESHOLD
                prediction_result_int = 1 if is_person_detected else 0
                
                if IS_INTERACTIVE_RUN:
                    print(f"{TermColors.OKBLUE}  重建誤差: {reconstruction_error:.4f} (閾值: {RECONSTRUCTION_ERROR_THRESHOLD:.4f}){TermColors.ENDC}", file=sys.stderr)

            except Exception as e_predict:
                if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}  LSTM AE 預測過程中出錯: {e_predict}{TermColors.ENDC}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr if IS_INTERACTIVE_RUN else sys.stdout)
                print("error") # 輸出錯誤信號給 index.js
                sys.exit(1) # 嚴重錯誤，退出
        else: # feature_sequence_for_model is None
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}  特徵序列準備失敗，無法預測。{TermColors.ENDC}", file=sys.stderr)
            print("buffering") # 可能是數據還不夠，或中間有NaN
            sys.exit(0) # 正常退出，等待下次調用
            
    else: # 緩存未滿
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}  包級別特徵緩存 {len(data_buffer_packet_features)} < {WINDOW_SIZE}，輸出 buffering...{TermColors.ENDC}", file=sys.stderr)
        print("buffering") # 輸出 buffering 信號給 index.js
        sys.exit(0) # 正常退出，等待下次調用

    # 輸出最終預測結果 (0 或 1) 到 stdout 給 index.js
    print(str(prediction_result_int))
    if IS_INTERACTIVE_RUN:
        pred_text = "有人 (Person)" if prediction_result_int == 1 else "無人 (No Person)"
        pred_color = TermColors.FAIL if prediction_result_int == 1 else TermColors.OKGREEN
        print(f"{TermColors.BOLD}{pred_color}--- predict.py 最終預測: {pred_text} ({prediction_result_int}) ---{TermColors.ENDC}\n", file=sys.stderr)

if __name__ == '__main__':
    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.HEADER}=== CSI Occupancy Predictor (CLI - LSTM Autoencoder) ==={TermColors.ENDC}", file=sys.stderr)
        print(f"  模型文件: {final_tflite_path_to_load if loaded_interpreter else '未加載'}", file=sys.stderr)
        print(f"  元數據文件: {METADATA_FILE_PKL}", file=sys.stderr)
        print(f"  數據緩存文件: {DATA_BUFFER_FILE}", file=sys.stderr)
        print(f"  窗口大小 (來自元數據): {WINDOW_SIZE}", file=sys.stderr)
        print(f"  使用特徵 (來自元數據): {LSTM_INPUT_FEATURES}", file=sys.stderr)
        print(f"  特徵數量 (來自元數據): {N_LSTM_FEATURES}", file=sys.stderr)
        print(f"  重建誤差閾值 (來自元數據): {RECONSTRUCTION_ERROR_THRESHOLD:.4f}\n", file=sys.stderr)
    try:
        main()
    except SystemExit: # 允許腳本通過 sys.exit(0) 正常退出
        pass
    except Exception as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: predict.py 主函數未處理異常: {e}{TermColors.ENDC}", file=sys.stderr)
        else: sys.stderr.write(f"FatalError: predict.py main unhandled exception: {e}\n")
        traceback.print_exc(file=sys.stderr if IS_INTERACTIVE_RUN else sys.stdout)
        print("error"); # 向 index.js 報告嚴重錯誤
        sys.exit(1)
