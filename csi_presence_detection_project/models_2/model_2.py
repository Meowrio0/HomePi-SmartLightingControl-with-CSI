import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
import warnings
from tqdm import tqdm
import traceback
import shutil
import sys

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

# --- GPU 設定 ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{TermColors.OKGREEN}GPU is available: {len(gpus)} GPU(s) detected{TermColors.ENDC}")
    except RuntimeError as e: print(f"{TermColors.FAIL}Error setting up GPU: {e}{TermColors.ENDC}")
else: print(f"{TermColors.WARNING}No GPU detected, using CPU{TermColors.ENDC}")
warnings.filterwarnings('ignore')

# --- 常數與超參數 ---
WALLHACK_LLTF_SUBCARRIER_INDICES = list(range(6, 32)) + list(range(33, 59))
N_SUBCARRIERS_FOR_MODEL = 52

SAMPLING_RATE_HZ = 16
print(f"{TermColors.OKBLUE}[INFO] Using SAMPLING_RATE_HZ: {SAMPLING_RATE_HZ} Hz{TermColors.ENDC}")

WINDOW_DURATION_SEC = 1.5
WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
STEP_SIZE = WINDOW_SIZE // 2
SEQUENCE_LENGTH = WINDOW_SIZE
print(f"{TermColors.OKBLUE}[INFO] Based on {SAMPLING_RATE_HZ} Hz and {WINDOW_DURATION_SEC}s duration, SEQUENCE_LENGTH (WINDOW_SIZE) is now: {SEQUENCE_LENGTH}{TermColors.ENDC}")

N_EPOCHS_AE = 350
BATCH_SIZE_AE = 32
LSTM_UNITS_AE_L1 = 64
LSTM_UNITS_AE_L2 = 32
LEARNING_RATE_AE = 1e-4
L2_LAMBDA_AE = 1e-5

##### ***** MODIFICATION: 移除 _diff 特徵 ***** #####
PACKET_LEVEL_BASE_FEATURES_FOR_STATS = ['csi_amp_std']
PACKET_LEVEL_DIFF_FEATURES_FOR_STATS = [] # 設置為空列表，表示不使用差分特徵
BASE_FEATURES_FOR_WINDOW_STATS = PACKET_LEVEL_BASE_FEATURES_FOR_STATS + PACKET_LEVEL_DIFF_FEATURES_FOR_STATS
print(f"{TermColors.OKBLUE}[INFO] Using features for window stats: {BASE_FEATURES_FOR_WINDOW_STATS}{TermColors.ENDC}")
##### ***** END MODIFICATION ***** #####
CLASS_NAMES_BINARY = ['No Person (0)', 'Person (1)']

def parse_wallhack_csi_string_to_amplitudes(csi_str):
    try:
        values_str = csi_str.strip('[]').split(',')
        if len(values_str) % 2 != 0: return None
        complex_csi_full = []
        for i in range(0, len(values_str), 2):
            try:
                img = float(values_str[i])
                real = float(values_str[i+1])
                complex_csi_full.append(complex(real, img))
            except ValueError: return None
        csi_complex_array_full = np.array(complex_csi_full)
        if not WALLHACK_LLTF_SUBCARRIER_INDICES: return None
        required_len = max(WALLHACK_LLTF_SUBCARRIER_INDICES) + 1
        if len(csi_complex_array_full) < required_len: return None
        selected_amplitudes = np.abs(csi_complex_array_full[WALLHACK_LLTF_SUBCARRIER_INDICES])
        return selected_amplitudes if len(selected_amplitudes) == N_SUBCARRIERS_FOR_MODEL else None
    except Exception as e:
        return None

def load_data_binary(file_paths_to_load, for_debug_limit_packets=False):
    all_packets_data = []
    if not file_paths_to_load: print("Warning: No file paths provided to load_data_binary."); return pd.DataFrame()
    print(f"Loading data from {len(file_paths_to_load)} specified CSV files...")
    
    if not hasattr(load_data_binary, 'debug_packet_printed_formal_no_person'):
        load_data_binary.debug_packet_printed_formal_no_person = False

    for file_path in tqdm(file_paths_to_load, desc="Processing CSV files"):
        if not os.path.exists(file_path): print(f"Warning: File {file_path} does not exist, skipping."); continue
        
        if "no_person" in os.path.basename(file_path).lower():
            file_id_scenario = "scenario_no_person"
        elif "person" in os.path.basename(file_path).lower():
            file_id_scenario = "scenario_person"
        else:
            file_id_scenario = os.path.basename(file_path).replace('.csv', '')
        
        try:
            df_file_packets = pd.read_csv(file_path, usecols=['data', 'class'])
            for _, row in df_file_packets.iterrows():
                selected_amplitudes = parse_wallhack_csi_string_to_amplitudes(row['data'])
                if selected_amplitudes is not None:
                    label_raw = int(row['class']); binary_label = 1 if label_raw > 0 else 0
                    packet_features = {'label': binary_label, 'scenario': file_id_scenario}
                    valid_packet = True
                    val_csi_amp_std = np.nan
                    for base_feat_name in PACKET_LEVEL_BASE_FEATURES_FOR_STATS: # 現在只會是 ['csi_amp_std']
                        if base_feat_name == 'csi_amp_std':
                            val_csi_amp_std = np.std(selected_amplitudes)
                            val = val_csi_amp_std
                        else:
                            print(f"Warning: Unknown base feature '{base_feat_name}'."); val = np.nan
                        if np.isnan(val): valid_packet = False; break
                        packet_features[base_feat_name] = val
                    
                    if not valid_packet: continue
                    
                    if not load_data_binary.debug_packet_printed_formal_no_person and file_id_scenario == "scenario_no_person":
                        print(f"\n{TermColors.HEADER}[MODEL.PY FORMAL RUN - LOAD_DATA_BINARY - 第一个有效 'no_person' 包]{TermColors.ENDC}")
                        print(f"  原始CSI字符串 (data列, 前60字符): {str(row['data'])[:60]}...")
                        print(f"  parse_wallhack_csi_string_to_amplitudes (52个選定振幅) (前5个): {selected_amplitudes[:5]}")
                        print(f"  此包的 csi_amp_std: {val_csi_amp_std:.4f}")
                        load_data_binary.debug_packet_printed_formal_no_person = True 
                        
                    all_packets_data.append(packet_features)
        except Exception as e: print(f"Error processing file {file_path}: {e}"); traceback.print_exc()
             
    if not all_packets_data: print("Error: No valid packet data loaded."); return pd.DataFrame()
    df_all_packets = pd.DataFrame(all_packets_data)
    print(f"Total valid packets loaded: {len(df_all_packets)}")
    if len(df_all_packets) > 0 : print(f"Label distribution: {df_all_packets['label'].value_counts(normalize=True).sort_index()}")
    return df_all_packets

def prepare_sequence_data(df_processed_packets, sequence_length=SEQUENCE_LENGTH, step_size=STEP_SIZE, for_debug_limit_sequences=False):
    sequences = []; sequence_labels = []; group_ids_for_sequences = []
    print("Creating sequence data...")
    if df_processed_packets.empty: print("Warning: Input DataFrame is empty."); return np.array([]), np.array([]), np.array([])
    
    actual_base_features = [f for f in PACKET_LEVEL_BASE_FEATURES_FOR_STATS if f in df_processed_packets.columns] # Should be ['csi_amp_std']
    # PACKET_LEVEL_DIFF_FEATURES_FOR_STATS is empty, so actual_diff_features will be empty
    actual_diff_features = [f'{feat}_diff' for feat in actual_base_features if f'{feat}_diff' in df_processed_packets.columns] # This will be empty
    actual_window_features = actual_base_features + actual_diff_features # Should be just ['csi_amp_std']
    
    if not actual_base_features: print("Error: No base features in DataFrame for sequence prep."); return np.array([]), np.array([]), np.array([])
    
    # The loop for calculating diff features will not run if PACKET_LEVEL_DIFF_FEATURES_FOR_STATS is empty
    if PACKET_LEVEL_DIFF_FEATURES_FOR_STATS: # This condition is now false
        for col_name in actual_base_features:
            diff_col_name = f'{col_name}_diff'
            if diff_col_name in actual_window_features: # This check will prevent error if diff_col_name not in df
                 df_processed_packets[diff_col_name] = df_processed_packets.groupby('scenario')[col_name].diff().fillna(0)
    
    if not hasattr(prepare_sequence_data, 'debug_sequence_printed_formal'):
        prepare_sequence_data.debug_sequence_printed_formal = False

    for scenario_id, group_df in tqdm(df_processed_packets.groupby('scenario'), desc="Generating sequences"):
        if len(group_df) < sequence_length:
            print(f"Warning: Scenario {scenario_id} has only {len(group_df)} packets, less than sequence_length {sequence_length}. Skipping.")
            continue
        
        feature_cols = [col for col in actual_window_features if col in group_df.columns] # Should now only contain 'csi_amp_std'
        if not feature_cols : 
            print(f"Warning: No valid feature columns selected for scenario {scenario_id} based on actual_window_features: {actual_window_features}. Skipping."); continue
        if len(feature_cols) != len(actual_window_features): 
            print(f"Warning: Features mismatch in {scenario_id}. Expected based on actual_window_features: {actual_window_features}, Found in group_df: {feature_cols}. Skipping."); continue
        
        for i in range(0, len(group_df) - sequence_length + 1, step_size):
            window = group_df.iloc[i:i + sequence_length]
            sequence_feature_values = window[feature_cols].values # Shape will be (SEQUENCE_LENGTH, 1)
            if np.isnan(sequence_feature_values).any() or np.isinf(sequence_feature_values).any(): continue
            
            if not prepare_sequence_data.debug_sequence_printed_formal and scenario_id == "scenario_no_person":
                print(f"\n{TermColors.HEADER}[MODEL.PY FORMAL RUN - PREPARE_SEQUENCE_DATA - 第一个 'no_person' 序列 (特徵: {feature_cols})]{TermColors.ENDC}")
                print(f"  序列形狀: {sequence_feature_values.shape}") # Should be (24, 1)
                print(f"  序列內容 (前3行):\n{sequence_feature_values[:3]}")
                prepare_sequence_data.debug_sequence_printed_formal = True
            
            sequences.append(sequence_feature_values)
            sequence_labels.append(np.argmax(np.bincount(window['label'].astype(int))))
            group_ids_for_sequences.append(scenario_id)

    if not sequences: print("Warning: No sequences created."); return np.array([]), np.array([]), np.array([])
    X = np.array(sequences); y = np.array(sequence_labels); groups = np.array(group_ids_for_sequences)
    print(f"Created {len(sequences)} sequences with shape {X.shape}, and {len(groups)} group IDs.") # X.shape should be (num_sequences, SEQUENCE_LENGTH, 1)
    return X, y, groups

def build_lstm_autoencoder(input_shape, lstm_units_l1=LSTM_UNITS_AE_L1, lstm_units_l2=LSTM_UNITS_AE_L2, l2_lambda=L2_LAMBDA_AE):
    n_features = input_shape[1] # Will be 1
    sequence_len = input_shape[0]
    inputs = Input(shape=input_shape)
    encoded = LSTM(lstm_units_l1, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(lstm_units_l2, activation='tanh', return_sequences=False, kernel_regularizer=regularizers.l2(l2_lambda))(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = RepeatVector(sequence_len)(encoded)
    decoded = LSTM(lstm_units_l2, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(l2_lambda))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(lstm_units_l1, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(l2_lambda))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = TimeDistributed(Dense(n_features, activation='linear'))(decoded) # Output n_features will also be 1
    autoencoder = Model(inputs, decoded)
    optimizer = Adam(learning_rate=LEARNING_RATE_AE, clipnorm=1.0)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    print("LSTM Autoencoder model compiled with tanh activation, BatchNormalization, and gradient clipping.")
    return autoencoder

def calculate_reconstruction_errors(X_true, X_pred):
    if X_true.shape != X_pred.shape:
        min_timesteps = min(X_true.shape[1], X_pred.shape[1])
        min_features = min(X_true.shape[2], X_pred.shape[2])
        print(f"Warning: Shape mismatch for error calculation. Using common shape (:, :{min_timesteps}, :{min_features})")
        return np.mean(np.square(X_true[:, :min_timesteps, :min_features] - X_pred[:, :min_timesteps, :min_features]), axis=(1, 2))
    return np.mean(np.square(X_true - X_pred), axis=(1, 2))

# --- Main Execution ---
def main():
    global scaler 
    start_time = time.time()
    tf.keras.backend.clear_session(); print(f"{TermColors.OKBLUE}TF Keras backend session cleared.{TermColors.ENDC}") # 使用 TermColors
    colab_base_path = '/content/' # 或者您實際存放數據的路徑
    if "google.colab" not in sys.modules and not os.path.exists(colab_base_path):
        colab_base_path = './'; print(f"{TermColors.WARNING}Warning: Not in Colab, using '{colab_base_path}' as base.{TermColors.ENDC}")
    os.makedirs(colab_base_path, exist_ok=True)

    # ***** MODIFICATION: 定義多個 "no_person" 文件 *****
    # 【【重要】請確保以下文件名與您上傳到 Colab /content/ 目錄下的文件名完全一致】
    no_person_csv_files_list = [
        "no_person_udp_16hz.csv",           # 您第一個2小時的 "沒人" 數據
        "no_person_udp_16hz-2.csv"          # 您新加入的第二個2小時 "沒人" 數據
        # 如果還有更多，繼續添加
    ]
    no_person_file_paths = [os.path.join(colab_base_path, f) for f in no_person_csv_files_list]
    
    # 【【重要】請確保以下文件名與您上傳到 Colab /content/ 目錄下的文件名完全一致】
    PERSON_CSV_FILENAME = "person_udp_30hz.csv" 
    person_file_path = os.path.join(colab_base_path, PERSON_CSV_FILENAME)
    
    print(f"{TermColors.OKBLUE}[INFO] Using 'No Person' data files: {no_person_csv_files_list}{TermColors.ENDC}")
    print(f"{TermColors.OKBLUE}[INFO] Using 'Person' data file: {PERSON_CSV_FILENAME}{TermColors.ENDC}")

    all_check_files = no_person_file_paths + [person_file_path] 
    # ***** END MODIFICATION *****

    DEBUG_PIPELINE_MODE = False # 【確保為 False 以進行正式訓練】
    if DEBUG_PIPELINE_MODE:
        # 這部分邏輯在正式訓練時不會執行
        print(f"{TermColors.WARNING}[INFO] 調試模式啟用（不應在此正式運行中出現）。{TermColors.ENDC}")
    else:
        print(f"{TermColors.OKGREEN}[INFO] 正式運行模式啟用 (僅使用 csi_amp_std 特徵，並使用更新的 '沒人' 數據集)：將處理所有數據並完整訓練模型。{TermColors.ENDC}")

    # 檢查所有需要的文件是否存在
    files_missing = False
    for f_path in all_check_files: 
        if not os.path.exists(f_path):
            print(f"{TermColors.FAIL}Error: Data file {f_path} not found.{TermColors.ENDC}")
            files_missing = True
            if "google.colab" in sys.modules: # 僅在Colab環境中創建虛擬文件
                print(f"Creating dummy file for {f_path} to allow script structure testing...")
                os.makedirs(os.path.dirname(f_path), exist_ok=True) # 確保目錄存在
                dummy_header = "data,class\n"; csi_vals = ["1"] * (N_SUBCARRIERS_FOR_MODEL * 2)
                csi_str_part = ",".join(csi_vals)
                file_label = 0 if "no_person" in os.path.basename(f_path).lower() else 1
                dummy_row = f"\"[{csi_str_part}]\",{file_label}\n"
                with open(f_path, "w") as f_dummy: f_dummy.write(dummy_header); f_dummy.write(dummy_row * (SEQUENCE_LENGTH + STEP_SIZE + 50))
            # else: # 在非Colab環境，如果文件缺失則直接提示錯誤 (此處不return，讓後續files_missing處理)
            #     print("Please ensure the data files are present in the correct location.")
    
    if files_missing and not ("google.colab" in sys.modules): # 如果不在Colab且文件缺失，則不繼續
        print(f"{TermColors.FAIL}One or more critical data files are missing. Please check paths and upload files. Exiting.{TermColors.ENDC}")
        return
    elif files_missing and ("google.colab" in sys.modules):
        print(f"{TermColors.WARNING}Warning: One or more data files were not found and dummy files were created. Training will proceed with dummy data for those missing files.{TermColors.ENDC}")

    # ***** MODIFICATION: 修正打印語句並正確傳遞文件列表 *****
    print(f"{TermColors.OKBLUE}--- Step 1: Loading and preparing 'No Person' data (from {len(no_person_file_paths)} files: {', '.join(no_person_csv_files_list)}) ---{TermColors.ENDC}")
    df_no_person_raw = load_data_binary(no_person_file_paths, for_debug_limit_packets=DEBUG_PIPELINE_MODE) 
    # ***** END MODIFICATION *****
    
    if df_no_person_raw.empty: print(f"{TermColors.FAIL}Error: No data loaded from 'No Person' files. Exiting.{TermColors.ENDC}"); return
    
    # 假設所有 no_person 文件都應該是 label 0，並且可以歸為一個 scenario
    # 如果您的CSV文件本身就有正確的 'label' 和 'scenario'，則以下兩行可能不需要或需要調整
    # 例如，您可以通過文件名判斷並在 load_data_binary 中已設定了 scenario
    # 如果 load_data_binary 中已根據文件名設定 label 和 scenario，這裡就不需要強制覆蓋
    # df_no_person_raw['label'] = 0 
    # df_no_person_raw['scenario'] = "scenario_no_person_combined" # 給合併後的數據一個統一的 scenario 名稱，以便 prepare_sequence_data 正確分組

    X_np_all, y_np_all, _ = prepare_sequence_data(df_no_person_raw, for_debug_limit_sequences=DEBUG_PIPELINE_MODE)
    if X_np_all.size == 0: print(f"{TermColors.FAIL}No 'No Person' sequences created from files: {no_person_csv_files_list}. Exiting.{TermColors.ENDC}"); return # 使用列表名
    
    # --- 正式模式下的數據分割 ---
    stratify_y_np = y_np_all if np.unique(y_np_all).size > 1 else None
    X_ae_train_np, X_temp_np, y_ae_train_np, y_temp_np = train_test_split(
        X_np_all, y_np_all, test_size=0.3, random_state=42, shuffle=True, stratify=stratify_y_np)
    X_val_for_thresh_np, X_test_final_np, y_val_for_thresh_np, y_test_final_np = [np.array([])]*4
    if len(X_temp_np) < 2: # 至少需要2個樣本才能進行下一次分割
        if len(X_temp_np) == 1: X_val_for_thresh_np = X_temp_np; y_val_for_thresh_np = y_temp_np
        print(f"{TermColors.WARNING}Warning: Not enough 'No Person' samples in temp set ({len(X_temp_np)}) for further val/test split.{TermColors.ENDC}")
    else:
        stratify_y_temp_np = y_temp_np if np.unique(y_temp_np).size > 1 else None
        X_val_for_thresh_np, X_test_final_np, y_val_for_thresh_np, y_test_final_np = train_test_split(
            X_temp_np, y_temp_np, test_size=0.5, random_state=42, shuffle=True, stratify=stratify_y_temp_np)
    print(f"AE Training 'No Person' sequences: {len(X_ae_train_np)}")
    print(f"Threshold Validation 'No Person' sequences: {len(X_val_for_thresh_np)}")
    print(f"Final Test 'No Person' sequences: {len(X_test_final_np)}")

    print(f"\n{TermColors.OKBLUE}--- Step 2: Loading and preparing 'Person' data (from {PERSON_CSV_FILENAME}) ---{TermColors.ENDC}")
    df_person_raw = load_data_binary([person_file_path], for_debug_limit_packets=False) 
    if df_person_raw.empty: print(f"{TermColors.FAIL}Error: No data loaded from '{PERSON_CSV_FILENAME}'. Exiting.{TermColors.ENDC}"); return
    # 確保 'person' 數據的標籤和 scenario 被正確處理
    if 'label' in df_person_raw.columns and not (df_person_raw['label'] == 1).all():
        print(f"{TermColors.WARNING}Warning: '{PERSON_CSV_FILENAME}' contains labels other than 1. Filtering for label 1.{TermColors.ENDC}")
        df_person_raw = df_person_raw[df_person_raw['label'] == 1].copy()
        if df_person_raw.empty: print(f"{TermColors.FAIL}Error: After filtering '{PERSON_CSV_FILENAME}' for label 1, no data remains. Exiting.{TermColors.ENDC}"); return
    elif 'label' not in df_person_raw.columns and not df_person_raw.empty:
        print(f"{TermColors.WARNING}Warning: 'label' missing in {PERSON_CSV_FILENAME}, assuming all data is 'Person' (label 1).{TermColors.ENDC}")
        df_person_raw['label'] = 1
        # 確保 person 數據也有 scenario，如果 load_data_binary 中沒有根據文件名設置的話
        if 'scenario' not in df_person_raw.columns:
            df_person_raw['scenario'] = "scenario_person" # 或者從文件名推斷
    elif df_person_raw.empty: print(f"{TermColors.FAIL}Error: No data loaded from '{PERSON_CSV_FILENAME}'. Exiting.{TermColors.ENDC}"); return
    
    X_p_all, y_p_all, _ = prepare_sequence_data(df_person_raw, for_debug_limit_sequences=False)
    if X_p_all.size == 0: print(f"{TermColors.FAIL}No 'Person' sequences created from {PERSON_CSV_FILENAME}. Exiting.{TermColors.ENDC}"); return
    
    X_val_p_orig, X_test_p_orig, y_val_p_orig, y_test_p_orig = [np.array([])]*4
    if len(X_p_all) < 2:
        if len(X_p_all) == 1: X_val_p_orig = X_p_all; y_val_p_orig = y_p_all
        print(f"{TermColors.WARNING}Warning: Very few 'Person' samples ({len(X_p_all)}). 'Test' set for P might be empty.{TermColors.ENDC}")
    else:
        stratify_y_p_all = y_p_all if np.unique(y_p_all).size > 1 else None
        X_val_p_orig, X_test_p_orig, y_val_p_orig, y_test_p_orig = train_test_split(
            X_p_all, y_p_all, test_size=0.5, random_state=42, shuffle=True, stratify=stratify_y_p_all)
    print(f"Total 'Person' sequences from {PERSON_CSV_FILENAME}: {len(X_p_all)}")
    print(f"Threshold Validation 'Person' sequences: {len(X_val_p_orig)}")
    print(f"Final Test 'Person' sequences: {len(X_test_p_orig)}")

    print(f"\n{TermColors.OKBLUE}--- Step 3: Combining Validation and Test sets ---{TermColors.ENDC}")
    X_val_combined_orig, y_val_combined = (np.array([]), np.array([]))
    if X_val_for_thresh_np.size > 0 and X_val_p_orig.size > 0:
        X_val_combined_orig = np.concatenate((X_val_for_thresh_np, X_val_p_orig), axis=0)
        y_val_combined = np.concatenate((y_val_for_thresh_np, y_val_p_orig), axis=0)
    elif X_val_for_thresh_np.size > 0:
        X_val_combined_orig, y_val_combined = X_val_for_thresh_np, y_val_for_thresh_np
        print(f"{TermColors.WARNING}Warning: Validation set for thresholding contains only 'No Person' data.{TermColors.ENDC}")
    elif X_val_p_orig.size > 0:
        X_val_combined_orig, y_val_combined = X_val_p_orig, y_val_p_orig
        print(f"{TermColors.WARNING}Warning: Validation set for thresholding contains only 'Person' data.{TermColors.ENDC}")
    else: 
        print(f"{TermColors.FAIL}Error: Threshold validation data is empty for both classes. Exiting.{TermColors.ENDC}"); return
    if X_val_combined_orig.size > 0: X_val_combined_orig, y_val_combined = shuffle(X_val_combined_orig, y_val_combined, random_state=42)
    
    X_test_combined_orig, y_test_combined = (np.array([]), np.array([]))
    if X_test_final_np.size > 0 and X_test_p_orig.size > 0:
        X_test_combined_orig = np.concatenate((X_test_final_np, X_test_p_orig), axis=0)
        y_test_combined = np.concatenate((y_test_final_np, y_test_p_orig), axis=0)
    elif X_test_final_np.size > 0 : 
        X_test_combined_orig, y_test_combined = X_test_final_np, y_test_final_np
        print(f"{TermColors.WARNING}Warning: Test set only 'No Person'.{TermColors.ENDC}")
    elif X_test_p_orig.size > 0 : 
        X_test_combined_orig, y_test_combined = X_test_p_orig, y_test_p_orig
        print(f"{TermColors.WARNING}Warning: Test set only 'Person'.{TermColors.ENDC}")
    # Else, if both are empty, X_test_combined_orig remains empty, handled later.
    if X_test_combined_orig.size > 0: X_test_combined_orig, y_test_combined = shuffle(X_test_combined_orig, y_test_combined, random_state=42)

    if X_ae_train_np.size == 0: print(f"{TermColors.FAIL}AE training data empty. Exiting.{TermColors.ENDC}"); return
    print(f"AE Training: {len(X_ae_train_np)}, Threshold Val: {len(X_val_combined_orig)} (NP:{sum(y_val_combined==0)}, P:{sum(y_val_combined==1)}), Final Test: {len(X_test_combined_orig)} (NP:{sum(y_test_combined==0 if y_test_combined.size >0 else 0)}, P:{sum(y_test_combined==1 if y_test_combined.size >0 else 0)})")

    print(f"\n{TermColors.OKBLUE}--- Step 4: Normalizing data ---{TermColors.ENDC}")
    scaler = StandardScaler() 
    if X_ae_train_np.ndim != 3 or X_ae_train_np.shape[0] == 0 or X_ae_train_np.shape[2] == 0:
        print(f"{TermColors.FAIL}Error: X_ae_train_np (shape: {X_ae_train_np.shape}) is not suitable for scaler fitting. Exiting.{TermColors.ENDC}"); return
    n_samples_ae, n_timesteps_ae, n_features_ae = X_ae_train_np.shape
    X_ae_train_norm = scaler.fit_transform(X_ae_train_np.reshape(-1, n_features_ae)).reshape(n_samples_ae, n_timesteps_ae, n_features_ae)

    if X_ae_train_norm.size > 0 and ( (not hasattr(prepare_sequence_data, 'debug_sequence_printed_formal') or not prepare_sequence_data.debug_sequence_printed_formal)):
        if hasattr(load_data_binary, 'debug_packet_printed_formal_no_person') and load_data_binary.debug_packet_printed_formal_no_person and \
           hasattr(prepare_sequence_data, 'debug_sequence_printed_formal') and prepare_sequence_data.debug_sequence_printed_formal :
            print(f"\n{TermColors.HEADER}[MODEL.PY FORMAL RUN - MAIN - 第一个訓練序列 (已標準化, 特徵數: {n_features_ae})]{TermColors.ENDC}")
            print(f"  (此處 scaler 已對 X_ae_train_np 執行 fit_transform)")
            print(f"  標準化後序列形狀: {X_ae_train_norm[0].shape}")
            print(f"  標準化後序列內容 (前3行):\n{X_ae_train_norm[0][:3]}")
            
    X_val_combined_norm = np.array([])
    if X_val_combined_orig.size > 0:
        if X_val_combined_orig.ndim == 3 and X_val_combined_orig.shape[2] == n_features_ae :
             X_val_combined_norm = scaler.transform(X_val_combined_orig.reshape(-1, n_features_ae)).reshape(X_val_combined_orig.shape)
        else:
            print(f"{TermColors.WARNING}Warning: X_val_combined_orig shape {X_val_combined_orig.shape} or feature dimension {X_val_combined_orig.shape[2] if X_val_combined_orig.ndim == 3 else 'N/A'} incompatible with n_features_ae ({n_features_ae}). Skipping scaling validation data.{TermColors.ENDC}")
    
    X_test_combined_norm = np.array([])
    if X_test_combined_orig.size > 0:
         if X_test_combined_orig.ndim == 3 and X_test_combined_orig.shape[2] == n_features_ae :
            X_test_combined_norm = scaler.transform(X_test_combined_orig.reshape(-1, n_features_ae)).reshape(X_test_combined_orig.shape)
         else:
            print(f"{TermColors.WARNING}Warning: X_test_combined_orig shape {X_test_combined_orig.shape} or feature dimension {X_test_combined_orig.shape[2] if X_test_combined_orig.ndim == 3 else 'N/A'} incompatible with n_features_ae ({n_features_ae}). Skipping scaling test data.{TermColors.ENDC}")

    print(f"AE Train norm stats: min={np.min(X_ae_train_norm):.4f}, max={np.max(X_ae_train_norm):.4f}, mean={np.mean(X_ae_train_norm):.4f}, std={np.std(X_ae_train_norm):.4f}")
    if np.isinf(X_ae_train_norm).any() or np.isnan(X_ae_train_norm).any(): print("FATAL: NaN/Inf in X_ae_train_norm. Exiting."); return

    print(f"\n{TermColors.OKBLUE}--- Step 5: Building and training LSTM Autoencoder ---{TermColors.ENDC}")
    tf.keras.backend.clear_session()
    input_shape_ae_for_meta = (SEQUENCE_LENGTH, n_features_ae) 
    autoencoder_model = build_lstm_autoencoder(input_shape_ae_for_meta)
    print("Model Summary (for training):") 
    autoencoder_model.summary()
    callbacks_ae = [ EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)]
    history_ae = autoencoder_model.fit( X_ae_train_norm, X_ae_train_norm, epochs=N_EPOCHS_AE,
        batch_size=BATCH_SIZE_AE, shuffle=True, validation_split=0.2, callbacks=callbacks_ae, verbose=1 )
    if "DISPLAY" in os.environ or "google.colab" in sys.modules:
        if 'loss' in history_ae.history and 'val_loss' in history_ae.history:
            plt.figure(); plt.plot(history_ae.history['loss'], label='AE Train Loss'); plt.plot(history_ae.history['val_loss'], label='AE Val Loss')
            plt.title('Autoencoder Training History'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.savefig(os.path.join(colab_base_path, 'training_history.png')); plt.show()


    print(f"\n{TermColors.OKBLUE}--- Step 6: Determining reconstruction error threshold ---{TermColors.ENDC}")
    optimal_threshold = 0.1 # Default fallback
    if X_val_combined_norm.size == 0: # 檢查驗證集是否為空
        print(f"{TermColors.WARNING}Warning: Validation set (X_val_combined_norm) is empty. Cannot determine threshold. Using default: {optimal_threshold}{TermColors.ENDC}")
    elif len(np.unique(y_val_combined)) < 2 : # 檢查驗證集標籤是否至少有兩類
        print(f"{TermColors.WARNING}Warning: Validation set for thresholding has only one class. Cannot determine optimal threshold via ROC. Using 95th percentile of 'No Person' errors if available, or default.{TermColors.ENDC}")
        # 嘗試基於 "No Person" 數據的誤差設定閾值
        X_val_no_person_errors = val_reconstruction_errors[y_val_combined==0] if 'val_reconstruction_errors' in locals() and y_val_combined.size > 0 else np.array([])
        if X_val_no_person_errors.size > 0:
            optimal_threshold = np.percentile(X_val_no_person_errors, 99) # 例如，取99百分位
            print(f"Optimal threshold set to 99th percentile of 'No Person' validation errors: {optimal_threshold:.6f}")
        else:
            print(f"No 'No Person' validation errors available to set percentile threshold. Using default: {optimal_threshold}")
    else: # 正常情況，驗證集包含兩類數據
        X_val_reconstructed = autoencoder_model.predict(X_val_combined_norm)
        val_reconstruction_errors = calculate_reconstruction_errors(X_val_combined_norm, X_val_reconstructed)
        fpr, tpr, thresholds_roc = roc_curve(y_val_combined, val_reconstruction_errors)
        if len(thresholds_roc) <= 1 :
            print(f"{TermColors.WARNING}Warning: Not enough distinct error values for ROC. Using median error of anomalies or 95th percentile as fallback.{TermColors.ENDC}")
            if sum(y_val_combined==1) > 0 : optimal_threshold = np.median(val_reconstruction_errors[y_val_combined==1])
            elif val_reconstruction_errors.size > 0 : optimal_threshold = np.percentile(val_reconstruction_errors, 95)
            else : print(f"{TermColors.FAIL}Error: val_reconstruction_errors is empty, cannot set threshold.{TermColors.ENDC}");
        else: optimal_idx_roc = np.argmax(tpr - fpr); optimal_threshold = thresholds_roc[optimal_idx_roc]
        print(f"Optimal threshold from ROC curve (Youden's J): {optimal_threshold:.6f}")
        
        if ("DISPLAY" in os.environ or "google.colab" in sys.modules) and val_reconstruction_errors.size > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(val_reconstruction_errors[y_val_combined==0], color="blue", label="No Person (Normal)", kde=True, stat="density", element="step", common_norm=False)
            sns.histplot(val_reconstruction_errors[y_val_combined==1], color="red", label="Person (Anomaly)", kde=True, stat="density", element="step", common_norm=False)
            plt.axvline(optimal_threshold, color='green', ls='--', lw=2, label=f'Optimal Threshold = {optimal_threshold:.4f}');
            plt.title('Reconstruction Error Distribution on Validation Set'); plt.xlabel('Mean Squared Error'); plt.ylabel('Density'); plt.legend(); plt.savefig(os.path.join(colab_base_path, 'reconstruction_error_distribution.png')); plt.show()
            
            if len(np.unique(y_val_combined)) > 1: # PR Curve
                precisions, recalls, thresholds_pr = precision_recall_curve(y_val_combined, val_reconstruction_errors)
                if thresholds_pr.size > 0:
                    plt.figure(figsize=(8,6)); plt.plot(thresholds_pr, precisions[:-1], "b--", label="Precision"); plt.plot(thresholds_pr, recalls[:-1], "g-", label="Recall")
                    plt.title("Precision-Recall vs Threshold"); plt.xlabel("Threshold"); plt.legend(); plt.grid(True); plt.savefig(os.path.join(colab_base_path, 'precision_recall_curve.png')); plt.show()

    final_metrics = {}
    if X_test_combined_norm.size > 0 and y_test_combined.size > 0 and len(np.unique(y_test_combined)) > 1 :
        print(f"\n{TermColors.OKBLUE}--- Step 7: Evaluating on Final Test Set ---{TermColors.ENDC}")
        X_test_reconstructed = autoencoder_model.predict(X_test_combined_norm)
        test_reconstruction_errors = calculate_reconstruction_errors(X_test_combined_norm, X_test_reconstructed)
        y_pred_test = (test_reconstruction_errors > optimal_threshold).astype(int)
        final_metrics = {'accuracy': accuracy_score(y_test_combined, y_pred_test), 'precision': precision_score(y_test_combined, y_pred_test, zero_division=0),
                         'recall': recall_score(y_test_combined, y_pred_test, zero_division=0), 'f1': f1_score(y_test_combined, y_pred_test, zero_division=0),
                         'threshold': optimal_threshold}
        print(f"Final Test Metrics: Accuracy={final_metrics['accuracy']:.4f}, Precision={final_metrics['precision']:.4f}, Recall={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}")
        print("Classification Report on Test Set:"); print(classification_report(y_test_combined, y_pred_test, target_names=CLASS_NAMES_BINARY, zero_division=0, labels=np.arange(len(CLASS_NAMES_BINARY))))
        cm_test = confusion_matrix(y_test_combined, y_pred_test, labels=np.arange(len(CLASS_NAMES_BINARY)))
        if "DISPLAY" in os.environ or "google.colab" in sys.modules:
            plt.figure(figsize=(8,6)); sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_BINARY, yticklabels=CLASS_NAMES_BINARY)
            plt.title('Test Set Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.savefig(os.path.join(colab_base_path, 'confusion_matrix.png')); plt.show()
    else:
        print(f"{TermColors.WARNING}Final test set (X_test_combined_norm) empty or single class. Skipping final evaluation.{TermColors.ENDC}")
        final_metrics = {'threshold': optimal_threshold, 'note': 'Test set empty or single class'}

    print(f"\n{TermColors.OKBLUE}--- Step 8: Saving Autoencoder model (TFLite) and Metadata (PKL) ---{TermColors.ENDC}")
    base_model_filename_stem = f'csi_lstm_ae_S{int(SAMPLING_RATE_HZ)}hz_L{int(SEQUENCE_LENGTH)}_NO_DIFF_MORE_NP_DATA' 
    
    keras_export_filename = os.path.join(colab_base_path, base_model_filename_stem + '.keras') 
    tflite_float32_filename = os.path.join(colab_base_path, base_model_filename_stem + '_float32.tflite')
    metadata_filename = os.path.join(colab_base_path, base_model_filename_stem + '_metadata.pkl')
    
    try:
        if autoencoder_model is None: 
             print(f"{TermColors.FAIL}[ERROR] autoencoder_model is None before saving. This should not happen.{TermColors.ENDC}")
             return

        autoencoder_model.save(keras_export_filename) 
        print(f"Keras Autoencoder model saved in Keras format to: {keras_export_filename}")

        print("Preparing for TFLite conversion by rebuilding model on CPU if necessary...")
        trained_weights = autoencoder_model.get_weights()
        
        tf.keras.backend.clear_session() 
        print("Keras session cleared for CPU model rebuilding.")
        with tf.device('/cpu:0'): 
            print("Rebuilding model on CPU for TFLite conversion...")
            cpu_autoencoder_model = build_lstm_autoencoder(input_shape_ae_for_meta) 
            cpu_autoencoder_model.set_weights(trained_weights)
            print("Weights loaded into CPU-rebuilt model.")
        
        print("Converting CPU-rebuilt model to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(cpu_autoencoder_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model_content = converter.convert()
        with open(tflite_float32_filename, 'wb') as f: f.write(tflite_model_content)
        print(f"Float32 TFLite Autoencoder model saved to: {tflite_float32_filename}")

        model_metadata_package = {
            'tflite_model_colab_path': os.path.basename(tflite_float32_filename),
            'keras_model_colab_path': os.path.basename(keras_export_filename), 
            'feature_names': BASE_FEATURES_FOR_WINDOW_STATS, 
            'sequence_length': int(SEQUENCE_LENGTH),
            'n_features': int(input_shape_ae_for_meta[1]), 
            'class_names': CLASS_NAMES_BINARY,
            'reconstruction_error_threshold': float(optimal_threshold),
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'final_eval_metrics': final_metrics, 
            'sampling_rate_hz_used': SAMPLING_RATE_HZ,
            'scaler_object': scaler 
        }
        with open(metadata_filename, 'wb') as file: pickle.dump(model_metadata_package, file)
        print(f"Model metadata (including scaler and threshold) saved to: {metadata_filename}")
        if 'scaler_object' in model_metadata_package and model_metadata_package['scaler_object'] is not None:
            print(f"{TermColors.OKGREEN}{TermColors.BOLD}重要：元數據中已成功包含 'scaler_object'。{TermColors.ENDC}")
        else:
            print(f"{TermColors.FAIL}{TermColors.BOLD}嚴重錯誤：未能將 'scaler_object' 保存到元數據中！{TermColors.ENDC}")

    except Exception as e_export: print(f"Error during model export: {e_export}"); traceback.print_exc()

    end_time = time.time(); execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
if __name__ == "__main__":
    main()
    
if __name__ == "__main__":
    main()
