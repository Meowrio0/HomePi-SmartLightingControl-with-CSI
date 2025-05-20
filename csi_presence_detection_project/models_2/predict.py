import pandas as pd
import numpy as np
import pickle
import json
import time
import os
import sys
import traceback
from collections import deque # 用於管理數據緩衝區

# 在導入 TensorFlow 之前設置日誌級別，以減少啟動時的 INFO 日誌
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
import tensorflow as tf 

# ANSI Escape Codes for colors
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

IS_INTERACTIVE_RUN = sys.stdin.isatty() and __name__ == '__main__' 

# --- 0. Utility Functions (CSI解析) ---
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

def calculate_amplitudes_from_iq_pairs(iq_array, meta_elements=0, current_expected_amplitude_len=64):
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

def select_subcarriers_from_esp32_data(amplitudes_raw_esp32):
    global N_SUBCARRIERS_FOR_MODEL, RAW_AMPLITUDE_LEN_FROM_ESP32, WALLHACK_LLTF_SUBCARRIER_INDICES
    if not isinstance(amplitudes_raw_esp32, np.ndarray):
        return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
    if len(amplitudes_raw_esp32) == RAW_AMPLITUDE_LEN_FROM_ESP32: 
        try:
            if not WALLHACK_LLTF_SUBCARRIER_INDICES: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
            if max(WALLHACK_LLTF_SUBCARRIER_INDICES) >= RAW_AMPLITUDE_LEN_FROM_ESP32: # 等於也會越界
                if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}SelectSubcarriers: Max index in WALLHACK_LLTF_SUBCARRIER_INDICES is out of bounds for raw amplitudes.{TermColors.ENDC}", file=sys.stderr)
                return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
                
            selected_amps = amplitudes_raw_esp32[WALLHACK_LLTF_SUBCARRIER_INDICES]
            if len(selected_amps) == N_SUBCARRIERS_FOR_MODEL: return selected_amps
            else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan) 
        except IndexError: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
        except ValueError: 
             if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}SelectSubcarriers: WALLHACK_LLTF_SUBCARRIER_INDICES might be empty.{TermColors.ENDC}", file=sys.stderr)
             return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan)
    else: return np.full(N_SUBCARRIERS_FOR_MODEL, np.nan) 

# --- 全局常量 ---
EXPECTED_INT_ARRAY_LEN = 128 
RAW_AMPLITUDE_LEN_FROM_ESP32 = (EXPECTED_INT_ARRAY_LEN - NUM_META_ELEMENTS) // 2 
WALLHACK_LLTF_SUBCARRIER_INDICES = list(range(6, 32)) + list(range(33, 59)) 
N_SUBCARRIERS_FOR_MODEL = 52 

SAMPLING_RATE_HZ = 16.0 
WINDOW_DURATION_SEC = 1.5 
_INITIAL_WINDOW_SIZE = int(SAMPLING_RATE_HZ * WINDOW_DURATION_SEC) 
WINDOW_SIZE = _INITIAL_WINDOW_SIZE 

PACKET_LEVEL_BASE_FEATURES_FOR_STATS = ['csi_amp_std'] 
LSTM_INPUT_FEATURES_DEFAULT = list(PACKET_LEVEL_BASE_FEATURES_FOR_STATS) 
N_LSTM_FEATURES_DEFAULT = len(LSTM_INPUT_FEATURES_DEFAULT)
CLASS_NAMES_BINARY_DEFAULT = ['No Person (0)', 'Person (1)'] 

LSTM_INPUT_FEATURES = list(LSTM_INPUT_FEATURES_DEFAULT) 
N_LSTM_FEATURES = N_LSTM_FEATURES_DEFAULT      
CLASS_NAMES_BINARY = list(CLASS_NAMES_BINARY_DEFAULT) 
RECONSTRUCTION_ERROR_THRESHOLD = 0.5 # 默認閾值，會被元數據覆蓋

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
DATA_BUFFER_FILE = os.path.join(SCRIPT_DIR, 'csi_data_buffer_lstm_ae.json') 

# --- 模型文件路徑 (請根據您樹莓派上的實際路徑進行調整) ---
MODEL_PARENT_DIR = "/home/linjianxun/HomePi-SmartLighting/homebridge-smart-light-csi/python/" # 您提供的路徑
# 【重要】以下文件名應指向您最新訓練的模型產出物
ACTUAL_METADATA_FILENAME = 'CSI LSTM AE Metadata.pkl' # 例如，最新的元數據文件名
ACTUAL_TFLITE_FILENAME_DEFAULT = 'CSI LSTM AE Model.tflite' # 例如，最新的TFLite文件名

METADATA_FILE_PKL = os.path.join(MODEL_PARENT_DIR, ACTUAL_METADATA_FILENAME)

# --- 全局變量，在初始化時加載 ---
loaded_interpreter = None
scaler = None
model_metadata = {}

def initialize_resources():
    global loaded_interpreter, scaler, model_metadata, WINDOW_SIZE, LSTM_INPUT_FEATURES, \
           N_LSTM_FEATURES, RECONSTRUCTION_ERROR_THRESHOLD, SAMPLING_RATE_HZ, \
           PACKET_LEVEL_BASE_FEATURES_FOR_STATS, CLASS_NAMES_BINARY

    # 嘗試設置TensorFlow日誌級別，以減少不必要的INFO輸出
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass # 如果 tf.get_logger() 在此版本的 TF Lite Runtime 中不可用，則忽略

    try:
        if not os.path.exists(METADATA_FILE_PKL):
            log_message = f"FatalError: 元數據文件 {METADATA_FILE_PKL} 未找到。"
            if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}{log_message}{TermColors.ENDC}", file=sys.stderr)
            else: sys.stderr.write(log_message + "\n")
            return False 
            
        with open(METADATA_FILE_PKL, 'rb') as f:
            model_metadata = pickle.load(f)
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE}從 {METADATA_FILE_PKL} 加載了元數據。{TermColors.ENDC}", file=sys.stderr)

        WINDOW_SIZE = int(model_metadata.get('sequence_length', _INITIAL_WINDOW_SIZE))
        LSTM_INPUT_FEATURES = list(model_metadata.get('feature_names', LSTM_INPUT_FEATURES_DEFAULT))
        N_LSTM_FEATURES = int(model_metadata.get('n_features', len(LSTM_INPUT_FEATURES)))
        CLASS_NAMES_BINARY = list(model_metadata.get('class_names', CLASS_NAMES_BINARY_DEFAULT))
        RECONSTRUCTION_ERROR_THRESHOLD = float(model_metadata.get('reconstruction_error_threshold', RECONSTRUCTION_ERROR_THRESHOLD)) 
        SAMPLING_RATE_HZ = float(model_metadata.get('sampling_rate_hz_used', SAMPLING_RATE_HZ))

        if all(not f.endswith("_diff") for f in LSTM_INPUT_FEATURES):
            PACKET_LEVEL_BASE_FEATURES_FOR_STATS = list(LSTM_INPUT_FEATURES)
        else:
            PACKET_LEVEL_BASE_FEATURES_FOR_STATS = list(set([f.replace("_diff","") for f in LSTM_INPUT_FEATURES]))
        
        if 'scaler_object' in model_metadata and model_metadata['scaler_object'] is not None:
            scaler = model_metadata['scaler_object']
            if IS_INTERACTIVE_RUN: print(f"{TermColors.OKGREEN}{TermColors.BOLD}[DEBUG INIT] StandardScaler 從元數據加載成功。{TermColors.ENDC}", file=sys.stderr)
        else:
            data_was_scaled_in_training = model_metadata.get('data_scaled', True) 
            if data_was_scaled_in_training:
                log_message_scaler = "[CRITICAL WARNING] 元數據中未找到 'scaler_object'，但模型可能基於標準化數據訓練！"
                if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}{TermColors.BOLD}{log_message_scaler}{TermColors.ENDC}", file=sys.stderr)
                else: sys.stderr.write(log_message_scaler + "\n")
            else:
                if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}{TermColors.BOLD}[DEBUG INIT] 元數據指示不使用標準化。{TermColors.ENDC}", file=sys.stderr)
        
        # ***** 【【【 手動修改/覆蓋閾值請在此處進行實驗 】】】 *****
        # 從元數據加載的閾值是 RECONSTRUCTION_ERROR_THRESHOLD
        # 如果您想手動測試不同的閾值，可以在這裡覆蓋它：
        # 例如，如果您通過實時測試發現 1.2 更合適：
        # RECONSTRUCTION_ERROR_THRESHOLD = 1.2
        # if IS_INTERACTIVE_RUN:
        #     print(f"{TermColors.WARNING}{TermColors.BOLD}[DEBUG INIT] 手動覆蓋閾值為: {RECONSTRUCTION_ERROR_THRESHOLD:.6f}{TermColors.ENDC}", file=sys.stderr)
        # ***** 【手動修改/覆蓋閾值結束】 *****


        tflite_filename_from_meta = model_metadata.get('tflite_model_colab_path')
        actual_tflite_filename_to_load = os.path.basename(tflite_filename_from_meta) if tflite_filename_from_meta else ACTUAL_TFLITE_FILENAME_DEFAULT
        final_tflite_path_to_load = os.path.join(MODEL_PARENT_DIR, actual_tflite_filename_to_load)

        if not os.path.exists(final_tflite_path_to_load):
            log_message_tflite = f"FatalError: TFLite模型文件 {final_tflite_path_to_load} 未找到。"
            if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}{log_message_tflite}{TermColors.ENDC}", file=sys.stderr)
            else: sys.stderr.write(log_message_tflite + "\n")
            return False

        loaded_interpreter = tf.lite.Interpreter(model_path=final_tflite_path_to_load)
        loaded_interpreter.allocate_tensors()
        if IS_INTERACTIVE_RUN: 
            print(f"{TermColors.OKGREEN}TFLite Autoencoder模型 {final_tflite_path_to_load} 加載成功。{TermColors.ENDC}", file=sys.stderr)
        return True

    except Exception as e:
        log_message_init_err = f"FatalError: 初始化資源時出錯: {e}"
        if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}{log_message_init_err}{TermColors.ENDC}", file=sys.stderr)
        else: sys.stderr.write(log_message_init_err + "\n")
        traceback.print_exc(file=sys.stderr if IS_INTERACTIVE_RUN else sys.stdout)
        return False

def process_csi_packet_to_features(csi_raw_str_from_json):
    packet_features = {}
    csi_parsed_int_list = parse_csi_raw_to_int_array(csi_raw_str_from_json, EXPECTED_INT_ARRAY_LEN)
    csi_amplitudes_raw_esp32 = calculate_amplitudes_from_iq_pairs(csi_parsed_int_list, meta_elements=NUM_META_ELEMENTS, current_expected_amplitude_len=RAW_AMPLITUDE_LEN_FROM_ESP32)
    
    if isinstance(csi_amplitudes_raw_esp32, np.ndarray) and not np.isnan(csi_amplitudes_raw_esp32).all():
        selected_amplitudes = select_subcarriers_from_esp32_data(csi_amplitudes_raw_esp32)
        if isinstance(selected_amplitudes, np.ndarray) and len(selected_amplitudes) == N_SUBCARRIERS_FOR_MODEL and not np.isnan(selected_amplitudes).any():
            for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: 
                if base_feat_name == 'csi_amp_std':
                    packet_features['csi_amp_std'] = np.std(selected_amplitudes)
                else: packet_features[base_feat_name] = np.nan 
        else: 
            for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: packet_features[base_feat_name] = np.nan
    else: 
        for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: packet_features[base_feat_name] = np.nan
            
    packet_features['timestamp'] = time.time()
    return packet_features

def prepare_feature_sequence_for_model(window_packet_features_list):
    global WINDOW_SIZE, LSTM_INPUT_FEATURES, N_LSTM_FEATURES, scaler, model_metadata 

    if len(window_packet_features_list) != WINDOW_SIZE: return None
    df_window = pd.DataFrame(window_packet_features_list)
    if df_window.empty: return None

    for base_feat in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: 
        diff_col_name = f'{base_feat}_diff'
        if diff_col_name in LSTM_INPUT_FEATURES and base_feat in df_window.columns:
            df_window[diff_col_name] = df_window[base_feat].ffill().bfill().diff().fillna(0)
    
    for col in LSTM_INPUT_FEATURES:
        if col not in df_window.columns:
            df_window[col] = 0.0 

    try:
        sequence_data_raw = df_window[LSTM_INPUT_FEATURES].values 
    except KeyError:
        if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}Prepare sequence: 特徵提取失敗。{TermColors.ENDC}", file=sys.stderr)
        return None

    if np.isnan(sequence_data_raw).any() or np.isinf(sequence_data_raw).any():
        sequence_data_raw = np.nan_to_num(sequence_data_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(sequence_data_raw).any() or np.isinf(sequence_data_raw).any():
             return None

    if sequence_data_raw.shape != (WINDOW_SIZE, N_LSTM_FEATURES):
        return None

    sequence_data_scaled = np.copy(sequence_data_raw) 
    data_scaled_info = model_metadata.get('data_scaled', True) 

    if scaler is not None and data_scaled_info:
        try:
            sequence_data_scaled = scaler.transform(sequence_data_raw) 
        except Exception as e_scale:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}Prepare sequence: 標準化時出錯: {e_scale}。將使用原始序列。{TermColors.ENDC}", file=sys.stderr)
            sequence_data_scaled = sequence_data_raw 
    
    return np.reshape(sequence_data_scaled, (1, WINDOW_SIZE, N_LSTM_FEATURES))

def main():
    global RECONSTRUCTION_ERROR_THRESHOLD 

    data_buffer_packet_features = deque(maxlen=WINDOW_SIZE) 
    
    if os.path.exists(DATA_BUFFER_FILE):
        try:
            with open(DATA_BUFFER_FILE, 'r') as f:
                loaded_list = json.load(f)
                if isinstance(loaded_list, list):
                    data_buffer_packet_features.extend(loaded_list[-WINDOW_SIZE:]) 
            if IS_INTERACTIVE_RUN: print(f"{TermColors.OKCYAN}從 {DATA_BUFFER_FILE} 加載了 {len(data_buffer_packet_features)} 個緩存包特徵。{TermColors.ENDC}", file=sys.stderr)
        except Exception as e_load_buffer:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}警告: 讀取或解析緩存文件 {DATA_BUFFER_FILE} 失敗: {e_load_buffer}。將使用空緩存。{TermColors.ENDC}", file=sys.stderr)
            data_buffer_packet_features.clear()

    if IS_INTERACTIVE_RUN: 
        print(f"{TermColors.OKBLUE}  監聽 stdin... (由 index.js 提供JSON數據包，每行一個){TermColors.ENDC}", file=sys.stderr)
        print(f"{TermColors.OKBLUE}  當前使用閾值 (來自元數據或手動覆蓋): {RECONSTRUCTION_ERROR_THRESHOLD:.6f}{TermColors.ENDC}", file=sys.stderr)


    while True: 
        line = sys.stdin.readline()
        if not line: 
            if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE} predict.py: 從 stdin 讀到 EOF，準備退出。{TermColors.ENDC}", file=sys.stderr)
            break 

        try:
            data_json_from_stdin = json.loads(line)
            csi_raw_from_stdin = data_json_from_stdin.get('csi_raw')
        except json.JSONDecodeError:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: 無法解析來自 stdin 的 JSON: {line.strip()[:100]}...{TermColors.ENDC}", file=sys.stderr)
            print("error", flush=True) 
            continue 
        except KeyError:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: stdin 的 JSON 中缺少 'csi_raw' 鍵。{TermColors.ENDC}", file=sys.stderr)
            print("error", flush=True) 
            continue

        if csi_raw_from_stdin is None:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: 從 stdin 接收到的 'csi_raw' 為空。{TermColors.ENDC}", file=sys.stderr)
            print("buffering", flush=True) 
            continue

        new_packet_feature_sample = process_csi_packet_to_features(csi_raw_from_stdin)
        
        if all(feat in new_packet_feature_sample and not pd.isna(new_packet_feature_sample[feat]) for feat in PACKET_LEVEL_BASE_FEATURES_FOR_STATS):
            data_buffer_packet_features.append(new_packet_feature_sample)
        else:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING} predict.py: 計算出的基礎特徵無效，丟棄此包。{TermColors.ENDC}", file=sys.stderr)
            if len(data_buffer_packet_features) < WINDOW_SIZE:
                 print("buffering", flush=True)
            continue
        
        try:
            with open(DATA_BUFFER_FILE, 'w') as f: json.dump(list(data_buffer_packet_features), f) 
        except Exception as e_write_buffer:
            if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}警告: 保存數據緩存到 {DATA_BUFFER_FILE} 失敗: {e_write_buffer}{TermColors.ENDC}", file=sys.stderr)

        prediction_result_int = 0 
        
        if len(data_buffer_packet_features) == WINDOW_SIZE:
            if IS_INTERACTIVE_RUN: 
                if not hasattr(main, 'printed_buffer_full_once'): # 只打印一次緩衝區滿的信息
                    print(f"{TermColors.OKCYAN}  包級別特徵緩存首次已滿 ({WINDOW_SIZE} 個)，準備使用 LSTM AE 預測...{TermColors.ENDC}", file=sys.stderr)
                    setattr(main, 'printed_buffer_full_once', True)
            
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
                         print("error", flush=True) 
                         continue 

                    loaded_interpreter.set_tensor(input_details['index'], input_data_final)
                    loaded_interpreter.invoke()
                    reconstructed_sequence = loaded_interpreter.get_tensor(output_details['index'])
                    reconstruction_error = np.mean(np.square(input_data_final - reconstructed_sequence))
                    
                    is_person_detected = reconstruction_error > RECONSTRUCTION_ERROR_THRESHOLD
                    prediction_result_int = 1 if is_person_detected else 0
                    
                    if IS_INTERACTIVE_RUN:
                        # 減少stderr打印頻率
                        if not hasattr(main, 'last_stderr_print_time') or (time.time() - main.last_stderr_print_time > 1.0) or \
                           not hasattr(main, 'last_stderr_pred_val') or main.last_stderr_pred_val != prediction_result_int:
                            print(f"{TermColors.OKBLUE}  預測 ({time.strftime('%H:%M:%S')}) - ReconErr: {reconstruction_error:.4f} (Thresh: {RECONSTRUCTION_ERROR_THRESHOLD:.4f}) -> Out: {prediction_result_int}{TermColors.ENDC}", file=sys.stderr)
                            main.last_stderr_print_time = time.time()
                            main.last_stderr_pred_val = prediction_result_int
                except Exception as e_predict:
                    if IS_INTERACTIVE_RUN: print(f"{TermColors.FAIL}  LSTM AE 預測過程中出錯: {e_predict}{TermColors.ENDC}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr if IS_INTERACTIVE_RUN else sys.stdout)
                    print("error", flush=True) 
                    continue 
            else: 
                if IS_INTERACTIVE_RUN: print(f"{TermColors.WARNING}  特徵序列準備失敗，輸出 buffering...{TermColors.ENDC}", file=sys.stderr)
                print("buffering", flush=True) 
                continue
                
        else: 
            if IS_INTERACTIVE_RUN: 
                 current_buffer_len = len(data_buffer_packet_features)
                 if not hasattr(main, 'last_buffer_len_printed') or main.last_buffer_len_printed != current_buffer_len or \
                    (time.time() - getattr(main, 'last_buffer_print_time', 0) > 2.0) : # 每2秒或長度變化時打印
                     sys.stderr.write(f"\r  包級別特徵緩存 {current_buffer_len}/{WINDOW_SIZE}，輸出 buffering...      ")
                     sys.stderr.flush()
                     main.last_buffer_len_printed = current_buffer_len
                     main.last_buffer_print_time = time.time()
            print("buffering", flush=True) 
            continue 

        print(str(prediction_result_int), flush=True) 

        if IS_INTERACTIVE_RUN:
            pred_text = "有人 (Person)" if prediction_result_int == 1 else "無人 (No Person)"
            pred_color = TermColors.FAIL if prediction_result_int == 1 else TermColors.OKGREEN
            # 減少最終預測的打印頻率
            if not hasattr(main, 'last_final_pred_print_time') or (time.time() - main.last_final_pred_print_time > 2.0) or \
               not hasattr(main, 'last_final_pred_val') or main.last_final_pred_val != prediction_result_int:
                print(f"{TermColors.BOLD}{pred_color}--- predict.py 最終預測 (@{time.strftime('%H:%M:%S')}): {pred_text} ({prediction_result_int}) ---{TermColors.ENDC}", file=sys.stderr)
                # print(file=sys.stderr) # 移除這個額外的空行，讓輸出更緊湊
                main.last_final_pred_print_time = time.time()
                main.last_final_pred_val = prediction_result_int

    if IS_INTERACTIVE_RUN: print(f"\n{TermColors.OKBLUE}predict.py stdin 管道已關閉或接收到 Ctrl+C，正常結束。{TermColors.ENDC}", file=sys.stderr)

if __name__ == '__main__':
    init_success = initialize_resources() 

    if not init_success:
        if not IS_INTERACTIVE_RUN: print("critical_error_exit", flush=True)
        sys.exit(1) 

    if IS_INTERACTIVE_RUN:
        print(f"{TermColors.HEADER}=== CSI Occupancy Predictor (CLI - LSTM Autoencoder) - 初始化完畢 ==={TermColors.ENDC}", file=sys.stderr)
        print(f"  模型文件: {final_tflite_path_to_load}", file=sys.stderr) # 使用初始化後賦值的全局變量
        print(f"  元數據文件: {METADATA_FILE_PKL}", file=sys.stderr)
        print(f"  數據緩存文件: {DATA_BUFFER_FILE}", file=sys.stderr)
        print(f"  窗口大小 (來自元數據): {WINDOW_SIZE}", file=sys.stderr)
        print(f"  使用特徵 (來自元數據): {LSTM_INPUT_FEATURES}", file=sys.stderr)
        print(f"  特徵數量 (來自元數據): {N_LSTM_FEATURES}", file=sys.stderr)
        print(f"  當前使用重建誤差閾值: {RECONSTRUCTION_ERROR_THRESHOLD:.6f}\n", file=sys.stderr) # 顯示實際使用的閾值
    try:
        main() 
    except SystemExit: 
        if IS_INTERACTIVE_RUN: print(f"{TermColors.OKBLUE} predict.py 正常退出 (SystemExit)。{TermColors.ENDC}", file=sys.stderr)
    except KeyboardInterrupt:
        if IS_INTERACTIVE_RUN: print(f"\n{TermColors.FAIL} predict.py 被用戶 Ctrl+C 終止。{TermColors.ENDC}", file=sys.stderr)
    except Exception as e:
        error_message = f"FatalError: predict.py main unhandled exception: {e}"
        if IS_INTERACTIVE_RUN:
            print(f"{TermColors.FAIL}{error_message}{TermColors.ENDC}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write(error_message + "\n")
        
        print("critical_error_exit", flush=True)
        sys.exit(1)



