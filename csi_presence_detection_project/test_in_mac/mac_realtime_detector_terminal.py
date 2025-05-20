import pandas as pd
import numpy as np
import pickle
import socket
import json
import time
import os
import sys
import traceback
from collections import deque

import tensorflow as tf

# Define TermColors class before its first use
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

IS_INTERACTIVE_RUN_FOR_TF_CONFIG = sys.stdin.isatty() and __name__ == '__main__'
try:
    tf.config.set_visible_devices([], 'GPU') # 強制使用 CPU
    if IS_INTERACTIVE_RUN_FOR_TF_CONFIG: print(f"{TermColors.OKBLUE}[DEBUG TF CONFIG] 已成功設置 TensorFlow 不使用 GPU (強制 CPU)。{TermColors.ENDC}", file=sys.stderr)
except Exception as e_tf_config:
    if IS_INTERACTIVE_RUN_FOR_TF_CONFIG: print(f"{TermColors.FAIL}[DEBUG TF CONFIG] 設置 TensorFlow 不使用 GPU 時發生錯誤: {e_tf_config}{TermColors.ENDC}", file=sys.stderr)

def parse_csi_raw_to_int_array(csi_string, expected_length=None):
    if pd.isna(csi_string): return np.full(expected_length, np.nan) if expected_length is not None else []
    try:
        s = str(csi_string).strip()
        if s.startswith('[') and s.endswith(']'): s = s[1:-1]
        if not s: return np.full(expected_length, np.nan) if expected_length is not None else []
        csi_values = [int(x) for x in s.split(',')]
        if expected_length is not None and len(csi_values) != expected_length: return np.full(expected_length, np.nan)
        return np.array(csi_values)
    except Exception: return np.full(expected_length, np.nan) if expected_length is not None else []

NUM_META_ELEMENTS = 0
def calculate_amplitudes_from_iq_pairs(iq_array, meta_elements=0, current_expected_amplitude_len=64):
    if not isinstance(iq_array, np.ndarray) or pd.isna(iq_array).all() or len(iq_array) <= meta_elements:
        return np.full(current_expected_amplitude_len, np.nan)
    actual_iq_data = iq_array[meta_elements:]
    if len(actual_iq_data) == 0 or len(actual_iq_data) % 2 != 0:
        return np.full(current_expected_amplitude_len, np.nan)
    amplitudes = []
    for i in range(0, len(actual_iq_data), 2):
        I_val = actual_iq_data[i]; Q_val = actual_iq_data[i+1]
        if pd.isna(I_val) or pd.isna(Q_val): amplitudes.append(np.nan)
        else: amplitudes.append(np.sqrt(float(I_val)**2 + float(Q_val)**2))
    if len(amplitudes) != current_expected_amplitude_len: return np.full(current_expected_amplitude_len, np.nan)
    return np.array(amplitudes)

# --- 全局常量 (將從元數據或默認值初始化) ---
EXPECTED_INT_ARRAY_LEN = 128
RAW_AMPLITUDE_LEN_FROM_ESP32 = (EXPECTED_INT_ARRAY_LEN - NUM_META_ELEMENTS) // 2
WALLHACK_LLTF_SUBCARRIER_INDICES = []
WALLHACK_LLTF_SUBCARRIER_INDICES.extend(list(range(6, 32)))
WALLHACK_LLTF_SUBCARRIER_INDICES.extend(list(range(33, 59)))
N_SUBCARRIERS_FOR_MODEL = 52

SAMPLING_RATE_HZ = 16.0 
WINDOW_DURATION_SEC = 1.5
_INITIAL_WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
WINDOW_SIZE = _INITIAL_WINDOW_SIZE 

PACKET_LEVEL_BASE_FEATURES_FOR_STATS = ['csi_amp_std'] 
LSTM_INPUT_FEATURES_DEFAULT = list(PACKET_LEVEL_BASE_FEATURES_FOR_STATS) 
N_LSTM_FEATURES_DEFAULT = len(LSTM_INPUT_FEATURES_DEFAULT)
CLASS_NAMES_BINARY_DEFAULT = ['No Person (0)', 'Person (1)'] 

LSTM_INPUT_FEATURES = list(LSTM_INPUT_FEATURES_DEFAULT) 
N_LSTM_FEATURES = N_LSTM_FEATURES_DEFAULT      
CLASS_NAMES_BINARY = list(CLASS_NAMES_BINARY_DEFAULT) 

UDP_IP = "0.0.0.0"; UDP_PORT = 5555; BUFFER_SIZE_UDP = 2048

# ----- 【重要】請根據您的實際文件路徑和名稱修改以下常量 -----
MODEL_PARENT_DIR = "/Users/linjianxun/Desktop/csi_presence_detection_project/models"
# 【指定您最新訓練的單特徵模型的元數據文件名 (應包含 _NO_DIFF)】
ACTUAL_METADATA_FILENAME = 'csi_lstm_ae_S16hz_L24_NO_DIFF_metadata.pkl' 
# 【指定您最新訓練的單特徵模型的TFLite文件名 (應包含 _NO_DIFF)】
ACTUAL_TFLITE_FILENAME_DEFAULT = 'csi_lstm_ae_S16hz_L24_NO_DIFF_float32.tflite'
# ----- 【重要】請根據您的實際文件路徑和名稱修改以上常量 -----

METADATA_FILE_PKL = os.path.join(MODEL_PARENT_DIR, ACTUAL_METADATA_FILENAME)

FINAL_DECISION_INTERVAL_SEC = 2.0
RECONSTRUCTION_ERROR_THRESHOLD = 0.5 # 將由元數據更新

INITIAL_DETECTION_DELAY_SEC = 5.0
STATS_PRINT_INTERVAL_SEC = 30.0 # 每30秒打印一次統計

IS_INTERACTIVE_RUN = sys.stdin.isatty() and __name__ == '__main__'
loaded_interpreter = None
model_type = 'tflite'
model_metadata = {}
scaler = None

try:
    tf.keras.backend.clear_session()
    if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE}[DEBUG INIT] TensorFlow Keras backend session cleared.{TermColors.ENDC}", file=sys.stderr)

    if not os.path.exists(METADATA_FILE_PKL):
        print(f"{TermColors.FAIL}FatalError: 元數據文件 {METADATA_FILE_PKL} 未找到。請確保文件名和路徑正確。{TermColors.ENDC}", file=sys.stderr)
        sys.exit(1)
    with open(METADATA_FILE_PKL, 'rb') as f: model_metadata = pickle.load(f)
    if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE}從 {METADATA_FILE_PKL} 加載了元數據。{TermColors.ENDC}", file=sys.stderr)

    WINDOW_SIZE = int(model_metadata.get('sequence_length', _INITIAL_WINDOW_SIZE))
    
    LSTM_INPUT_FEATURES = list(model_metadata.get('feature_names', LSTM_INPUT_FEATURES_DEFAULT))
    N_LSTM_FEATURES = int(model_metadata.get('n_features', len(LSTM_INPUT_FEATURES))) 
    
    # 根據元數據實際加載的特徵名更新基礎特徵列表
    if all(not f.endswith("_diff") for f in LSTM_INPUT_FEATURES):
        PACKET_LEVEL_BASE_FEATURES_FOR_STATS = list(LSTM_INPUT_FEATURES)
    else:
        PACKET_LEVEL_BASE_FEATURES_FOR_STATS = list(set([f.replace("_diff","") for f in LSTM_INPUT_FEATURES]))
    if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}[DEBUG INIT] 基於元數據，將使用的基礎特徵 (PACKET_LEVEL_BASE_FEATURES_FOR_STATS): {PACKET_LEVEL_BASE_FEATURES_FOR_STATS}{TermColors.ENDC}", file=sys.stderr)

    CLASS_NAMES_BINARY = list(model_metadata.get('class_names', CLASS_NAMES_BINARY_DEFAULT)) 
    RECONSTRUCTION_ERROR_THRESHOLD = float(model_metadata.get('reconstruction_error_threshold', RECONSTRUCTION_ERROR_THRESHOLD))
    
    loaded_sampling_rate = model_metadata.get('sampling_rate_hz_used')
    if loaded_sampling_rate is not None:
        SAMPLING_RATE_HZ = float(loaded_sampling_rate)
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE}[DEBUG INIT] 從元數據更新 SAMPLING_RATE_HZ: {SAMPLING_RATE_HZ:.1f} Hz{TermColors.ENDC}", file=sys.stderr)
    elif IS_INTERACTIVE_RUN:
         print(f"{TermColors.WARNING}[DEBUG INIT] 元數據中未找到 'sampling_rate_hz_used'，將使用默認值: {SAMPLING_RATE_HZ:.1f} Hz{TermColors.ENDC}", file=sys.stderr)

    if 'scaler_object' in model_metadata and model_metadata['scaler_object'] is not None:
        scaler = model_metadata['scaler_object']
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKGREEN}{TermColors.BOLD}[DEBUG INIT] StandardScaler 從元數據加載成功。模型將使用標準化數據進行推斷。{TermColors.ENDC}", file=sys.stderr)
    else:
        data_was_scaled_in_training = model_metadata.get('data_scaled', True) 
        if data_was_scaled_in_training: 
            if IS_INTERACTIVE_RUN:
                print(f"{TermColors.FAIL}{TermColors.BOLD}[CRITICAL WARNING] 元數據中未找到 'scaler_object' 或其值為 None，但數據在訓練時可能已標準化！{TermColors.ENDC}", file=sys.stderr)
                print(f"{TermColors.FAIL}{TermColors.BOLD}推斷將使用【未標準化】的數據，這【極有可能】導致模型性能嚴重下降或完全錯誤。{TermColors.ENDC}", file=sys.stderr)
                print(f"{TermColors.FAIL}{TermColors.BOLD}請務必修改 model.py 以正確保存 scaler 並重新生成模型及元數據，或確認模型訓練時未使用標準化。{TermColors.ENDC}", file=sys.stderr)
        else: 
             if IS_INTERACTIVE_RUN:
                print(f"{TermColors.OKCYAN}{TermColors.BOLD}[DEBUG INIT] 元數據指示訓練數據未經標準化 ('data_scaled': False)。實時推斷也將不使用標準化。{TermColors.ENDC}", file=sys.stderr)


    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.OKBLUE}[DEBUG INIT] 最終使用WINDOW_SIZE: {WINDOW_SIZE}{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.OKBLUE}{TermColors.BOLD}[DEBUG INIT] 最終使用LSTM_INPUT_FEATURES (來自元數據或默認): {LSTM_INPUT_FEATURES}{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.OKBLUE}{TermColors.BOLD}[DEBUG INIT] 最終使用N_LSTM_FEATURES (來自元數據或計算): {N_LSTM_FEATURES}{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.OKBLUE}[DEBUG INIT] 最終使用CLASS_NAMES_BINARY: {CLASS_NAMES_BINARY}{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.OKBLUE}[DEBUG INIT] 最終使用RECONSTRUCTION_ERROR_THRESHOLD: {RECONSTRUCTION_ERROR_THRESHOLD:.6f}{TermColors.ENDC}", file=sys.stderr)
        if N_LSTM_FEATURES != len(LSTM_INPUT_FEATURES):
                 print(f"{TermColors.WARNING}[DEBUG INIT] 警告: n_features ({N_LSTM_FEATURES}) 與 feature_names 的長度 ({len(LSTM_INPUT_FEATURES)}) 不匹配!{TermColors.ENDC}", file=sys.stderr)

    tflite_filename_from_meta = model_metadata.get('tflite_model_colab_path') 
    actual_tflite_filename_to_load = os.path.basename(tflite_filename_from_meta) if tflite_filename_from_meta else ACTUAL_TFLITE_FILENAME_DEFAULT
    final_tflite_path_to_load = os.path.join(MODEL_PARENT_DIR, actual_tflite_filename_to_load)

    if not os.path.exists(final_tflite_path_to_load):
        print(f"{TermColors.FAIL}FatalError: TFLite模型文件 {final_tflite_path_to_load} 未找到。{TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.FAIL}請檢查 MODEL_PARENT_DIR ('{MODEL_PARENT_DIR}') 和元數據中的 'tflite_model_colab_path' 或 ACTUAL_TFLITE_FILENAME_DEFAULT 的設置。{TermColors.ENDC}", file=sys.stderr)
        sys.exit(1)
    
    loaded_interpreter = tf.lite.Interpreter(model_path=final_tflite_path_to_load)
    loaded_interpreter.allocate_tensors()
    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.OKGREEN}TFLite Autoencoder模型 {final_tflite_path_to_load} 加載成功。{TermColors.ENDC}", file=sys.stderr)
        input_details = loaded_interpreter.get_input_details()
        output_details = loaded_interpreter.get_output_details()
        print(f"TFLite Input Details: {input_details}", file=sys.stderr) # 確保打印到 stderr
        print(f"TFLite Output Details: {output_details}", file=sys.stderr) # 確保打印到 stderr

except Exception as e_load_main:
    print(f"{TermColors.FAIL}FatalError: 初始化或加載模型/元數據過程中發生嚴重錯誤: {e_load_main}{TermColors.ENDC}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

def select_subcarriers_from_esp32_data(amplitudes_raw_esp32):
    global N_SUBCARRIERS_FOR_MODEL, RAW_AMPLITUDE_LEN_FROM_ESP32, WALLHACK_LLTF_SUBCARRIER_INDICES
    if not isinstance(amplitudes_raw_esp32, np.ndarray):
        return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
    if len(amplitudes_raw_esp32) == RAW_AMPLITUDE_LEN_FROM_ESP32:
        try:
            if not WALLHACK_LLTF_SUBCARRIER_INDICES: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
            if max(WALLHACK_LLTF_SUBCARRIER_INDICES) < RAW_AMPLITUDE_LEN_FROM_ESP32:
                selected_amps = amplitudes_raw_esp32[WALLHACK_LLTF_SUBCARRIER_INDICES]
                if len(selected_amps) == N_SUBCARRIERS_FOR_MODEL: return selected_amps
                else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
            else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
        except IndexError: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
    else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)

def process_csi_to_packet_features_from_json(data_json):
    global N_SUBCARRIERS_FOR_MODEL, RAW_AMPLITUDE_LEN_FROM_ESP32, IS_INTERACTIVE_RUN, TermColors, PACKET_LEVEL_BASE_FEATURES_FOR_STATS, EXPECTED_INT_ARRAY_LEN, NUM_META_ELEMENTS
    print_debug_this_packet = IS_INTERACTIVE_RUN and not hasattr(process_csi_to_packet_features_from_json, 'debug_packet_printed')

    try: csi_raw_str = data_json['csi_raw']
    except (KeyError, ValueError, TypeError) as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}警告: JSON無效或csi_raw缺失: {e}{TermColors.ENDC}", file=sys.stderr)
        return None
    
    if print_debug_this_packet:
        print(f"{TermColors.HEADER}[TERMINAL.PY DEBUG PROCESS_CSI - 第一个UDP包]{TermColors.ENDC}", file=sys.stderr) # 移除 \n
        print(f"  原始CSI字符串 (JSON['csi_raw'], 前60字符): {csi_raw_str[:60]}...", file=sys.stderr)

    csi_parsed_int_list = parse_csi_raw_to_int_array(csi_raw_str, EXPECTED_INT_ARRAY_LEN)
    if print_debug_this_packet:
        print(f"  parse_csi_raw_to_int_array (長度: {len(csi_parsed_int_list)}, 前10个): {csi_parsed_int_list[:10]}", file=sys.stderr)

    csi_amplitudes_raw_esp32 = calculate_amplitudes_from_iq_pairs(csi_parsed_int_list, meta_elements=NUM_META_ELEMENTS, current_expected_amplitude_len=RAW_AMPLITUDE_LEN_FROM_ESP32)
    if print_debug_this_packet:
        print(f"  calculate_amplitudes_from_iq_pairs (64个原始振幅, 長度: {len(csi_amplitudes_raw_esp32)}, 前5个): {csi_amplitudes_raw_esp32[:5]}", file=sys.stderr)
    
    packet_features = {}
    if isinstance(csi_amplitudes_raw_esp32, np.ndarray) and not np.isnan(csi_amplitudes_raw_esp32).all():
        selected_amplitudes = select_subcarriers_from_esp32_data(csi_amplitudes_raw_esp32)
        if print_debug_this_packet:
            print(f"  select_subcarriers_from_esp32_data (52个選定振幅, 長度: {len(selected_amplitudes)}, 前5个): {selected_amplitudes[:5]}", file=sys.stderr)

        if isinstance(selected_amplitudes, np.ndarray) and len(selected_amplitudes) == N_SUBCARRIERS_FOR_MODEL and not np.isnan(selected_amplitudes).any():
            for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: 
                if base_feat_name == 'csi_amp_std': 
                    packet_features['csi_amp_std'] = np.std(selected_amplitudes)
                    if print_debug_this_packet:
                        print(f"  此包的 csi_amp_std: {packet_features['csi_amp_std']:.4f}", file=sys.stderr)
                else: packet_features[base_feat_name] = np.nan
        else:
            for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: packet_features[base_feat_name] = np.nan
            if print_debug_this_packet: print(f"  未能從選定振幅計算特徵 (selected_amplitudes 問題)", file=sys.stderr)
    else:
        for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: packet_features[base_feat_name] = np.nan
        if print_debug_this_packet: print(f"  未能從原始振幅計算特徵 (csi_amplitudes_raw_esp32 問題)", file=sys.stderr)

    if print_debug_this_packet:
        process_csi_to_packet_features_from_json.debug_packet_printed = True

    packet_features['timestamp'] = time.time()
    return packet_features

def prepare_lstm_input_sequence(window_packet_features_list):
    global WINDOW_SIZE, LSTM_INPUT_FEATURES, N_LSTM_FEATURES, IS_INTERACTIVE_RUN, TermColors
    func_name = "prepare_lstm_input_sequence"
    print_debug_this_sequence = IS_INTERACTIVE_RUN and not hasattr(prepare_lstm_input_sequence, 'debug_sequence_printed')

    if len(window_packet_features_list) != WINDOW_SIZE:
        return None
    df_window = pd.DataFrame(window_packet_features_list)
    if df_window.empty:
        return None
    
    temp_base_features = list(set([f.replace("_diff","") for f in LSTM_INPUT_FEATURES if f in df_window.columns]))

    for base_feat in temp_base_features: 
        diff_col_name = f'{base_feat}_diff'
        if diff_col_name in LSTM_INPUT_FEATURES and base_feat in df_window.columns: 
            df_window[diff_col_name] = df_window[base_feat].ffill().bfill().diff().fillna(0)
            
    for col in LSTM_INPUT_FEATURES: 
        if col not in df_window.columns:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}[DEBUG:{func_name}] 期望輸入特徵 '{col}' 缺失，填充0。{TermColors.ENDC}", file=sys.stderr)
            df_window[col] = 0.0
            
    try:
        sequence_data_raw = df_window[LSTM_INPUT_FEATURES].values
    except KeyError as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[DEBUG:{func_name}] 提取sequence_data時KeyError: {e}。期望: {LSTM_INPUT_FEATURES}，實際可用: {df_window.columns.tolist()}。返回 None。{TermColors.ENDC}", file=sys.stderr)
        return None

    if print_debug_this_sequence:
        print(f"{TermColors.HEADER}[TERMINAL.PY DEBUG PREPARE_LSTM_INPUT - 第一个序列 (未標準化, 特徵: {LSTM_INPUT_FEATURES})]{TermColors.ENDC}", file=sys.stderr) # 移除 \n
        print(f"  序列形狀: {sequence_data_raw.shape}", file=sys.stderr)
        print(f"  序列內容 (前3行):\n{sequence_data_raw[:3]}", file=sys.stderr)

    if np.isinf(sequence_data_raw).any() or np.isnan(sequence_data_raw).any():
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}[DEBUG:{func_name}] 序列數據包含Inf/NaN值，嘗試填充...{TermColors.ENDC}", file=sys.stderr)
        sequence_data_raw = np.nan_to_num(sequence_data_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isinf(sequence_data_raw).any() or np.isnan(sequence_data_raw).any():
             if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[DEBUG:{func_name}] 填充後序列數據仍包含Inf/NaN值。返回 None。{TermColors.ENDC}", file=sys.stderr)
             return None

    expected_shape = (WINDOW_SIZE, N_LSTM_FEATURES) 
    if sequence_data_raw.shape != expected_shape:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[DEBUG:{func_name}] 序列數據形狀 {sequence_data_raw.shape} 與預期 {expected_shape} (元數據N_LSTM_FEATURES={N_LSTM_FEATURES}) 不符。返回 None。{TermColors.ENDC}", file=sys.stderr)
        return None
    
    if print_debug_this_sequence:
        setattr(prepare_lstm_input_sequence, 'debug_sequence_printed', True)
    
    return np.reshape(sequence_data_raw, (1, WINDOW_SIZE, N_LSTM_FEATURES))

def main_loop():
    global loaded_interpreter, model_type, IS_INTERACTIVE_RUN, TermColors, WINDOW_SIZE, CLASS_NAMES_BINARY, model_metadata, FINAL_DECISION_INTERVAL_SEC, RECONSTRUCTION_ERROR_THRESHOLD, scaler, STATS_PRINT_INTERVAL_SEC, LSTM_INPUT_FEATURES, N_LSTM_FEATURES
    func_name = "main_loop"
    
    recent_short_term_errors = deque()
    last_final_decision_time = time.time()

    count_person_detected = 0
    count_no_person_detected = 0
    last_stats_print_time = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((UDP_IP, UDP_PORT))
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}[DEBUG:{func_name}] 準備就緒，監聽UDP端口 {UDP_PORT}...{TermColors.ENDC}", file=sys.stderr)
    except OSError as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[DEBUG:{func_name}] FatalError: 無法綁定端口 {UDP_PORT}: {e}{TermColors.ENDC}", file=sys.stderr)
        else: print(f"FatalError: Cannot bind port {UDP_PORT}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    data_buffer_packet_features_list = []
    last_header_print_time = time.time(); header_print_interval = 1.0
    packet_count_for_rate_calc = 0; rate_calc_start_time = time.time(); rate_display_interval = 5.0

    debug_scaled_sequence_printed = False
    debug_reconstruction_printed = False
    first_window_processed_for_debug_newline = False # 新標記

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE_UDP)
            message_str = data.decode('utf-8', errors='ignore')
            current_packet_arrival_time = time.time()
            packet_count_for_rate_calc +=1
            
            if IS_INTERACTIVE_RUN and (current_packet_arrival_time - last_header_print_time > header_print_interval):
                if not first_window_processed_for_debug_newline: # 確保在第一個窗口的詳細debug打印後有換行
                    if hasattr(process_csi_to_packet_features_from_json, 'debug_packet_printed') and \
                       hasattr(prepare_lstm_input_sequence, 'debug_sequence_printed') and \
                       debug_scaled_sequence_printed and \
                       debug_reconstruction_printed:
                        print(file=sys.stderr) # 在第一個完整debug塊後打印一個空行
                        first_window_processed_for_debug_newline = True

                print(f"{TermColors.HEADER}--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---{TermColors.ENDC}", file=sys.stderr) # 移除前導 \n
                elapsed_for_rate = current_packet_arrival_time - rate_calc_start_time
                if elapsed_for_rate >= rate_display_interval:
                    actual_rate = packet_count_for_rate_calc / elapsed_for_rate
                    print(f"{TermColors.OKCYAN}  過去 ~{elapsed_for_rate:.1f}s 平均接收速率: {actual_rate:.2f} Hz ({packet_count_for_rate_calc} 包){TermColors.ENDC}", file=sys.stderr)
                    packet_count_for_rate_calc = 0; rate_calc_start_time = current_packet_arrival_time
                last_header_print_time = current_packet_arrival_time
            
            try: data_json = json.loads(message_str)
            except json.JSONDecodeError:
                continue
            
            new_packet_feature = process_csi_to_packet_features_from_json(data_json)

            if new_packet_feature and all(feat in new_packet_feature and not np.isnan(new_packet_feature[feat]) for feat in PACKET_LEVEL_BASE_FEATURES_FOR_STATS):
                new_packet_feature['packet_timestamp'] = new_packet_feature['timestamp']
                data_buffer_packet_features_list.append(new_packet_feature)

                while len(data_buffer_packet_features_list) > WINDOW_SIZE:
                    data_buffer_packet_features_list.pop(0)
                
                if IS_INTERACTIVE_RUN and len(data_buffer_packet_features_list) < WINDOW_SIZE and not \
                   (hasattr(process_csi_to_packet_features_from_json, 'debug_packet_printed') and \
                    hasattr(prepare_lstm_input_sequence, 'debug_sequence_printed') and \
                    debug_scaled_sequence_printed and \
                    debug_reconstruction_printed): # 只有在所有debug打完之前才更新緩存狀態
                    sys.stderr.write(f"\r  包級別特徵緩存: {len(data_buffer_packet_features_list)}/{WINDOW_SIZE}      ")
                    sys.stderr.flush()

                if len(data_buffer_packet_features_list) == WINDOW_SIZE:
                    # 當緩衝區滿了，並且是第一次處理（即 process_csi 的 debug 標記已設定），則清掉 \r 行
                    if IS_INTERACTIVE_RUN and \
                       hasattr(process_csi_to_packet_features_from_json, 'debug_packet_printed') and \
                       not hasattr(main_loop, 'cleared_buffer_line_after_debug'):
                        sys.stderr.write("\n") 
                        setattr(main_loop, 'cleared_buffer_line_after_debug', True)
                        
                    lstm_input_sequence_raw = prepare_lstm_input_sequence(list(data_buffer_packet_features_list))
                    
                    if lstm_input_sequence_raw is not None:
                        if IS_INTERACTIVE_RUN and not hasattr(main_loop, 'feature_check_done'):
                            print(f"{TermColors.OKCYAN}[DEBUG MAIN_LOOP] 首次處理序列，元數據加載的 LSTM_INPUT_FEATURES: {LSTM_INPUT_FEATURES}, N_LSTM_FEATURES: {N_LSTM_FEATURES}{TermColors.ENDC}", file=sys.stderr)
                            print(f"{TermColors.OKCYAN}[DEBUG MAIN_LOOP] 當前序列的 shape[2] (特徵數): {lstm_input_sequence_raw.shape[2]}{TermColors.ENDC}", file=sys.stderr)
                            if lstm_input_sequence_raw.shape[2] != N_LSTM_FEATURES:
                                print(f"{TermColors.FAIL}[CRITICAL ERROR] 實時序列特徵數 ({lstm_input_sequence_raw.shape[2]}) 與元數據中 N_LSTM_FEATURES ({N_LSTM_FEATURES}) 不匹配！{TermColors.ENDC}", file=sys.stderr)
                            setattr(main_loop, 'feature_check_done', True)

                        lstm_input_sequence_scaled = np.copy(lstm_input_sequence_raw)
                        
                        if scaler is not None:
                            try:
                                original_shape = lstm_input_sequence_raw.shape
                                temp_sequence_reshaped = lstm_input_sequence_raw.reshape(original_shape[1], original_shape[2])
                                scaled_sequence_reshaped = scaler.transform(temp_sequence_reshaped)
                                lstm_input_sequence_scaled = scaled_sequence_reshaped.reshape(original_shape)
                                if IS_INTERACTIVE_RUN and not debug_scaled_sequence_printed:
                                    print(f"{TermColors.HEADER}[TERMINAL.PY DEBUG MAIN_LOOP - 第一个序列 (已標準化, 特徵: {LSTM_INPUT_FEATURES})]{TermColors.ENDC}", file=sys.stderr) # 移除 \n
                                    print(f"  標準化後序列形狀: {lstm_input_sequence_scaled.shape}", file=sys.stderr)
                                    print(f"  標準化後序列內容 (前3行):\n{lstm_input_sequence_scaled[0, :3]}", file=sys.stderr)
                                    debug_scaled_sequence_printed = True
                            except Exception as e_scale:
                                if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}    [數據準備] 標準化實時序列時出錯: {e_scale}。使用原始序列。{TermColors.ENDC}", file=sys.stderr)
                        elif IS_INTERACTIVE_RUN and not debug_scaled_sequence_printed:
                            print(f"{TermColors.HEADER}[TERMINAL.PY DEBUG MAIN_LOOP - 第一个序列 (由於Scaler缺失或元數據指示不縮放, 使用未標準化數據, 特徵: {LSTM_INPUT_FEATURES})]{TermColors.ENDC}", file=sys.stderr) # 移除 \n
                            print(f"  (未標準化)序列形狀: {lstm_input_sequence_scaled.shape}", file=sys.stderr)
                            print(f"  (未標準化)序列內容 (前3行):\n{lstm_input_sequence_scaled[0, :3]}", file=sys.stderr)
                            debug_scaled_sequence_printed = True

                        reconstruction_error = -1.0
                        if model_type == 'tflite':
                            input_details = loaded_interpreter.get_input_details()[0]
                            output_details = loaded_interpreter.get_output_details()[0]
                            
                            lstm_input_sequence_final = lstm_input_sequence_scaled
                            if lstm_input_sequence_final.dtype != input_details['dtype']:
                                lstm_input_sequence_final = lstm_input_sequence_final.astype(input_details['dtype'])
                            
                            if lstm_input_sequence_final.shape != tuple(input_details['shape']):
                                if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[ERROR] TFLite 輸入形狀不匹配。期望: {input_details['shape']}, 實際: {lstm_input_sequence_final.shape}{TermColors.ENDC}", file=sys.stderr)
                                continue 

                            loaded_interpreter.set_tensor(input_details['index'], lstm_input_sequence_final)
                            loaded_interpreter.invoke()
                            reconstructed_sequence = loaded_interpreter.get_tensor(output_details['index'])
                            reconstruction_error = np.mean(np.square(lstm_input_sequence_final - reconstructed_sequence))
                        
                        if reconstruction_error >= 0:
                            current_window_timestamp = data_buffer_packet_features_list[-1]['packet_timestamp']
                            recent_short_term_errors.append({'error': reconstruction_error, 'time': current_window_timestamp})
                            if IS_INTERACTIVE_RUN and not debug_reconstruction_printed:
                                print(f"{TermColors.HEADER}[TERMINAL.PY DEBUG MAIN_LOOP - 第一个重建誤差]{TermColors.ENDC}", file=sys.stderr) # 移除 \n
                                print(f"  重建誤差 (基於 {'已標準化' if scaler and model_metadata.get('data_scaled', True) else '未標準化'} 數據): {reconstruction_error:.4f}", file=sys.stderr)
                                debug_reconstruction_printed = True
            
            current_eval_time = time.time()
            if current_eval_time - last_final_decision_time >= FINAL_DECISION_INTERVAL_SEC:
                last_final_decision_time = current_eval_time
                cutoff_time = current_eval_time - FINAL_DECISION_INTERVAL_SEC
                while recent_short_term_errors and recent_short_term_errors[0]['time'] < cutoff_time:
                    recent_short_term_errors.popleft()

                should_print_final_decision = not IS_INTERACTIVE_RUN or \
                                             (hasattr(process_csi_to_packet_features_from_json, 'debug_packet_printed') and \
                                              hasattr(prepare_lstm_input_sequence, 'debug_sequence_printed') and \
                                              debug_scaled_sequence_printed and \
                                              debug_reconstruction_printed)
                
                if IS_INTERACTIVE_RUN and should_print_final_decision:
                    print(f"{TermColors.OKCYAN}[最終決策@{time.strftime('%H:%M:%S')}] Buffer中有效短期誤差數: {len(recent_short_term_errors)}{TermColors.ENDC}", file=sys.stderr)

                if recent_short_term_errors:
                    errors_in_buffer = [item['error'] for item in recent_short_term_errors]
                    aggregated_error = np.mean(errors_in_buffer) if errors_in_buffer else 0.0
                    min_err_in_buffer = np.min(errors_in_buffer) if errors_in_buffer else 0.0
                    max_err_in_buffer = np.max(errors_in_buffer) if errors_in_buffer else 0.0
                    is_person_finally_detected = aggregated_error > RECONSTRUCTION_ERROR_THRESHOLD
                    
                    if is_person_finally_detected:
                        count_person_detected += 1
                    else:
                        count_no_person_detected += 1

                    final_status_text = CLASS_NAMES_BINARY[1] if is_person_finally_detected else CLASS_NAMES_BINARY[0]
                    final_status_color = TermColors.FAIL if is_person_finally_detected else TermColors.OKGREEN
                    
                    if should_print_final_decision:
                        print(f"{TermColors.BOLD}{final_status_color}>>> [最終判斷] {final_status_text} "
                              f"(聚合誤差: {aggregated_error:.4f}, 閾值: {RECONSTRUCTION_ERROR_THRESHOLD:.4f}, "
                              f"短期誤差範圍: [{min_err_in_buffer:.4f}-{max_err_in_buffer:.4f}], "
                              f"基於過去{FINAL_DECISION_INTERVAL_SEC:.1f}秒內的{len(recent_short_term_errors)}個短期重建){TermColors.ENDC}",
                              file=sys.stderr if IS_INTERACTIVE_RUN else sys.stdout)
                        if not IS_INTERACTIVE_RUN: sys.stdout.flush()
                        if IS_INTERACTIVE_RUN: print(file=sys.stderr) # 在最終判斷後加一個空行

                elif IS_INTERACTIVE_RUN and should_print_final_decision:
                     print(f"{TermColors.WARNING}[最終決策@{time.strftime('%H:%M:%S')}] 誤差緩衝區為空，本次無最終判斷。{TermColors.ENDC}", file=sys.stderr)
                     if IS_INTERACTIVE_RUN: print(file=sys.stderr) # 也加一個空行
            
            current_time_for_stats = time.time()
            if IS_INTERACTIVE_RUN and STATS_PRINT_INTERVAL_SEC > 0 and \
               (current_time_for_stats - last_stats_print_time > STATS_PRINT_INTERVAL_SEC):
                print(f"{TermColors.OKCYAN}[統計數據@{time.strftime('%H:%M:%S')}] " # 移除前導 \n
                      f"有人判斷: {count_person_detected} 次, "
                      f"沒人判斷: {count_no_person_detected} 次{TermColors.ENDC}", file=sys.stderr)
                print(file=sys.stderr) # 在統計數據後加一個空行
                last_stats_print_time = current_time_for_stats

        except socket.timeout:
            continue
        except KeyboardInterrupt:
            print(f"\n{TermColors.FAIL}[DEBUG:{func_name}] 程序被用戶終止。{TermColors.ENDC}", file=sys.stderr)
            if IS_INTERACTIVE_RUN:
                print(f"{TermColors.HEADER}=== 最終統計數據 ==={TermColors.ENDC}", file=sys.stderr)
                print(f"{TermColors.OKGREEN}總計「有人」判斷: {count_person_detected} 次{TermColors.ENDC}", file=sys.stderr)
                print(f"{TermColors.OKBLUE}總計「沒人」判斷: {count_no_person_detected} 次{TermColors.ENDC}", file=sys.stderr)
            break
        except Exception as e:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}[DEBUG:{func_name}] 主循環發生未捕獲錯誤: {e}{TermColors.ENDC}", file=sys.stderr)
            else: print(f"ErrorLoop: {e}", file=sys.stderr)
            if IS_INTERACTIVE_RUN: traceback.print_exc(file=sys.stderr)
            if not IS_INTERACTIVE_RUN: print("0"); sys.stdout.flush()
            time.sleep(1)

    sock.close()
    if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}[DEBUG:{func_name}] UDP監聽已關閉。{TermColors.ENDC}", file=sys.stderr)

if __name__ == '__main__':
    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.HEADER}=== CSI 實時佔用檢測 (UDP - Autoencoder TFLite - 帶決策平滑) ==={TermColors.ENDC}", file=sys.stderr)
        print(f"  TFLite模型文件: {final_tflite_path_to_load} (存在: {os.path.exists(final_tflite_path_to_load)})", file=sys.stderr)
        print(f"  元數據 PKL: {METADATA_FILE_PKL} (存在: {os.path.exists(METADATA_FILE_PKL)})", file=sys.stderr)
        # ... (其他啟動信息打印) ...
        print(f"  重建誤差閾值 (來自元數據): {RECONSTRUCTION_ERROR_THRESHOLD:.4f}", file=sys.stderr) 
        print(file=sys.stderr) # 在初始信息塊後添加一個空行
        
        if INITIAL_DETECTION_DELAY_SEC > 0:
            print(f"{TermColors.WARNING}系統將在 {INITIAL_DETECTION_DELAY_SEC:.1f} 秒後開始偵測...{TermColors.ENDC}", file=sys.stderr)
            time.sleep(INITIAL_DETECTION_DELAY_SEC)
        print(f"{TermColors.OKGREEN}=== 開始實時檢測 ==={TermColors.ENDC}", file=sys.stderr)
        # print(file=sys.stderr) # 在 "開始實時檢測" 後不再加空行，讓第一個時間戳緊隨其後

    try:
        main_loop()
    except SystemExit:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE}[DEBUG] 程序正常退出 (SystemExit)。{TermColors.ENDC}", file=sys.stderr)
    except Exception as e:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}FatalError: terminal.py __main__ 未處理異常: {e}{TermColors.ENDC}", file=sys.stderr)
        else: print(f"FatalError: terminal.py main: {e}", file=sys.stderr)
        if IS_INTERACTIVE_RUN: traceback.print_exc(file=sys.stderr)
        if not IS_INTERACTIVE_RUN: print("0"); sys.stdout.flush()
        sys.exit(1)