import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import json # Not used in this version, can be removed if not needed elsewhere
import sys
from scipy.stats import gmean # For Spectral Flatness Measure

# 1. Load Data and Initial Inspection
plt.style.use('ggplot')
sns.set_style('whitegrid')

file_path = '/Users/linjianxun/Desktop/csi_presence_detection_project/datasets/processed_data/csi_merged.csv' # 請確認路徑
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"ERROR: File not found at '{file_path}'. Please check the path and name.")
    sys.exit()

print("--- Data Basic Information ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Numerical Features Statistics ---")
print(df.describe())

print("\n--- Categorical Features Statistics ---")
print(df.describe(include=['object', 'category']))

print("\n--- Missing Values Check ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# --- 2. Target Variable Analysis ---
print("\n--- Target Variable Analysis ---")

# Analyze occupancy
plt.figure(figsize=(8, 5))
sns.countplot(x='occupancy', data=df)
plt.title('Distribution of Occupancy')
plt.xlabel('Occupancy (0: Empty, 1: Presence)')
plt.ylabel('Count')
plt.show()
print("Occupancy Proportions:")
print(df['occupancy'].value_counts(normalize=True))

# Analyze scenario
plt.figure(figsize=(12, 7))
sns.countplot(y='scenario', data=df, order = df['scenario'].value_counts().index)
plt.title('Distribution of Scenarios')
plt.xlabel('Count')
plt.ylabel('Scenario')
plt.tight_layout()
plt.show()
print("\nScenario Proportions:")
print(df['scenario'].value_counts(normalize=True))

# Check consistency between scenario and occupancy (Adjusted Logic)
print("\n--- Consistency Check: Scenario vs Occupancy (Adjusted) ---")
empty_scenarios = [s for s in df['scenario'].unique() if str(s).startswith('empty')]
if empty_scenarios:
    for s in empty_scenarios:
        occupancy_values = df[df['scenario'] == s]['occupancy'].unique()
        print(f"Scenario '{s}' (expected empty) - Occupancy values found: {occupancy_values}")
        if not (len(occupancy_values) == 1 and occupancy_values[0] == 0):
            print(f"WARNING: Scenario '{s}' should have occupancy 0, but found {occupancy_values}!")
else:
    print("No scenarios starting with 'empty' found.")

presence_scenarios = [s for s in df['scenario'].unique() if str(s).startswith('presence')]
if presence_scenarios:
    for s in presence_scenarios:
        occupancy_values = df[df['scenario'] == s]['occupancy'].unique()
        print(f"Scenario '{s}' (expected presence) - Occupancy values found: {occupancy_values}")
        if not (len(occupancy_values) == 1 and occupancy_values[0] == 1):
            print(f"WARNING: Scenario '{s}' should have occupancy 1, but found {occupancy_values}!")
else:
    print("No scenarios starting with 'presence' found.")


# --- 3. Timestamp Analysis ---
print("\n--- Timestamp Analysis ---")
if 'local_timestamp' in df.columns:
    try:
        df['timestamp_dt'] = pd.to_datetime(df['local_timestamp'])
        print("Successfully converted 'local_timestamp' directly.")
    except (ValueError, TypeError, OverflowError):
        try:
            df['timestamp_dt'] = pd.to_datetime(df['local_timestamp'], unit='s')
            print("Interpreted 'local_timestamp' as Unix timestamp in seconds.")
        except (ValueError, TypeError, OverflowError):
            try:
                df['timestamp_dt'] = pd.to_datetime(df['local_timestamp'], unit='ms')
                print("Interpreted 'local_timestamp' as Unix timestamp in milliseconds.")
            except Exception as e_ts:
                print(f"Could not convert 'local_timestamp' to datetime: {e_ts}. Timestamp analysis might be affected.")
                df['timestamp_dt'] = pd.NaT
    
    if 'timestamp_dt' in df.columns and not df['timestamp_dt'].isnull().any() :
        df = df.sort_values(by='timestamp_dt').reset_index(drop=True)
        print(f"Data time range: From {df['timestamp_dt'].min()} To {df['timestamp_dt'].max()}")

        df['time_diff_seconds'] = df.groupby('scenario')['timestamp_dt'].diff().dt.total_seconds() # Group by scenario for diff
        print("\nStatistics for time differences between consecutive samples (seconds, within scenarios):")
        print(df['time_diff_seconds'].describe())

        valid_time_diffs = df['time_diff_seconds'].dropna()
        if not valid_time_diffs.empty:
            hist_range_upper = valid_time_diffs.quantile(0.995) if len(valid_time_diffs) > 1 else valid_time_diffs.iloc[0]
            if pd.isna(hist_range_upper) or hist_range_upper <= 0:
                 hist_range_upper = 1.0
            
            plt.figure(figsize=(10, 5))
            df['time_diff_seconds'].hist(bins=100, range=(0, hist_range_upper) )
            plt.title(f'Distribution of Sample Time Intervals (seconds - up to {hist_range_upper:.2f}s)')
            plt.xlabel('Time Interval (seconds)')
            plt.ylabel('Frequency')
            plt.show()

            # Calculate time gaps between SCENARIO BLOCKS, not just any large gap
            # This requires identifying the end of one scenario block and start of another based on original data sequence
            # For simplicity in EDA, we'll rely on visual inspection of the 'Scenario Data Points Over Time' plot
            # The original large_gaps logic might pick up gaps *within* a scenario if there was a recording hiccup.
            # True inter-scenario gaps are best seen by looking at sorted 'local_timestamp' before any scenario grouping.
            df_sorted_by_orig_ts = df.sort_values(by='local_timestamp').copy() # Assuming local_timestamp is the original int
            df_sorted_by_orig_ts['orig_time_diff'] = pd.to_datetime(df_sorted_by_orig_ts['local_timestamp'], unit='ms').diff().dt.total_seconds() # example unit
            
            # Adjust threshold for detecting gaps between blocks
            # This threshold needs careful thought; it should be larger than typical intra-scenario gaps
            # but smaller than intentional pauses between recordings.
            # Example: if scenarios are ~3min (180s) and sampling is ~40ms
            # a gap of > 1 second might indicate a block change.
            inter_block_gap_threshold = 1.0 # seconds
            large_gaps_between_blocks = df_sorted_by_orig_ts[df_sorted_by_orig_ts['orig_time_diff'] > inter_block_gap_threshold]
            print(f"\nDetected {len(large_gaps_between_blocks)} time gaps larger than {inter_block_gap_threshold:.2f} seconds (likely between DIFFERENT scenario blocks):")
            if not large_gaps_between_blocks.empty:
                print(large_gaps_between_blocks[['timestamp_dt', 'orig_time_diff', 'scenario']].head())
            else:
                print("No unusually large time gaps (suggesting block changes) detected with the current threshold on original timestamps.")


        plt.figure(figsize=(18, 8))
        unique_scenarios_sorted = sorted(df['scenario'].unique())
        for i, scenario_name in enumerate(unique_scenarios_sorted):
            scenario_data = df[df['scenario'] == scenario_name]
            plt.scatter(scenario_data['timestamp_dt'], [i] * len(scenario_data), label=scenario_name, s=5)
        plt.yticks(range(len(unique_scenarios_sorted)), unique_scenarios_sorted)
        plt.title('Scenario Data Points Over Time (Sorted by converted timestamp_dt)')
        plt.xlabel('Timestamp (converted)')
        plt.ylabel('Scenario')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("'local_timestamp' could not be properly converted to datetime or contains NaT values. Skipping detailed time analysis.")
else:
    print("'local_timestamp' column not found.")

# --- 4. Numerical Feature Analysis ---
print("\n--- Numerical Feature Analysis ---")
constant_cols_std_zero = []
if not df.empty:
    desc_stats = df.describe(include=np.number) # include only numerical for std
    if 'std' in desc_stats.index:
        constant_cols_std_zero = desc_stats.columns[desc_stats.loc['std'] == 0].tolist()
print(f"Columns with standard deviation of 0 (likely constant): {constant_cols_std_zero}")

# Primary numerical features from original data to analyze
numerical_features_to_plot = ['rssi'] # 'noise_floor', 'ampdu_cnt', 'sig_len' were often constant or less informative
numerical_features_to_plot = [col for col in numerical_features_to_plot if col in df.columns and col not in constant_cols_std_zero]

if numerical_features_to_plot:
    print(f"Analyzing primary numerical features: {numerical_features_to_plot}")
    for feature in numerical_features_to_plot:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.subplot(1, 3, 2)
        sns.boxplot(x='occupancy', y=feature, data=df)
        plt.title(f'{feature} vs. Occupancy')
        plt.subplot(1, 3, 3)
        sns.boxplot(x='scenario', y=feature, data=df)
        plt.title(f'{feature} vs. Scenario')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # --- NEW: RSSI Dynamic Characteristics ---
    if 'rssi' in df.columns and 'rssi' not in constant_cols_std_zero:
        print("\n--- NEW: RSSI Dynamic Characteristics ---")
        df['rssi_diff_within_scenario'] = df.groupby('scenario')['rssi'].diff()
        df['abs_rssi_diff_within_scenario'] = df['rssi_diff_within_scenario'].abs()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='scenario', y='abs_rssi_diff_within_scenario', data=df.dropna(subset=['abs_rssi_diff_within_scenario']))
        plt.title('Absolute RSSI Difference (within scenario) vs. Scenario')
        plt.ylabel('Abs(RSSI[t] - RSSI[t-1])')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='occupancy', y='abs_rssi_diff_within_scenario', data=df.dropna(subset=['abs_rssi_diff_within_scenario']))
        plt.title('Absolute RSSI Difference (within scenario) vs. Occupancy')
        plt.ylabel('Abs(RSSI[t] - RSSI[t-1])')
        plt.tight_layout()
        plt.show()
    # --- END NEW RSSI ---

    if len(numerical_features_to_plot) > 1: # Correlation matrix for selected features
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_features_to_plot].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix of: {", ".join(numerical_features_to_plot)}')
        plt.show()
else:
    print("No variable primary numerical features selected for plotting or available.")

# --- 5. Categorical Feature Analysis ---
print("\n--- Categorical Feature Analysis ---")
categorical_features = ['mac'] # MAC is often unique per device, might not be useful for general model
potential_cats_from_int = ['sig_mode', 'mcs', 'bandwidth', 'smoothing', 'not_sounding',
                           'aggregation', 'stbc', 'fec_coding', 'sgi', 'ant', 'rx_state',
                           'channel', 'secondary_channel']
for p_cat in potential_cats_from_int:
    if p_cat in df.columns and df[p_cat].nunique() < 20 : # Limit to features with few unique values
        if p_cat not in constant_cols_std_zero: # Exclude if identified as constant
             categorical_features.append(p_cat)
        df[p_cat] = df[p_cat].astype('category') # Convert to category type

if 'type' in df.columns and df['type'].nunique() == 1:
    print(f"Column 'type' has only one unique value: {df['type'].unique()[0]}. Skipping its detailed categorical plot.")
    if 'type' in categorical_features:
        categorical_features.remove('type')
categorical_features = sorted(list(set([col for col in categorical_features if col in df.columns]))) # Ensure unique and sorted

if categorical_features:
    print(f"Analyzing categorical features: {categorical_features}")
    for feature in categorical_features:
        if df[feature].nunique() < 30: # Plot if not too many unique values
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            order = df[feature].value_counts().index[:15] if not df.empty else []
            sns.countplot(y=feature, data=df, order=order)
            plt.title(f'Frequency of {feature} (Top 15)')
            
            plt.subplot(1, 2, 2)
            # Use hue_order to ensure consistent color mapping for occupancy
            hue_order = sorted(df['occupancy'].unique()) 
            top_categories = df[feature].value_counts().index[:5] if not df.empty else []
            temp_df_cat = df[df[feature].isin(top_categories)]
            sns.countplot(y=feature, hue='occupancy', data=temp_df_cat, order=top_categories, hue_order=hue_order)
            plt.title(f'{feature} vs. Occupancy (Top 5 cat.)')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Feature '{feature}' has too many unique values ({df[feature].nunique()}) for direct plotting. Showing top 5 value counts:")
            print(df[feature].value_counts().head())
else:
    print("No suitable categorical features selected for analysis.")

# --- 6. CSI Raw Data Analysis (Integrated Parsing and Amplitude Calculation) ---
print("\n--- CSI Raw Data Analysis ---")
csi_amplitude_columns = [] # Initialize, will be populated later

if 'csi_raw' not in df.columns:
    print("'csi_raw' column not found. Skipping CSI analysis.")
else:
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

    num_meta_elements = 0 # !!! USER ACTION REQUIRED !!!
    print(f"IMPORTANT: Assuming the first {num_meta_elements} integers in parsed 'csi_raw' are metadata and will be skipped for I/Q processing.")
    print("If this is incorrect, please adjust 'num_meta_elements' in the script.")

    def calculate_amplitudes_from_iq_pairs(iq_array, meta_elements=0):
        if not isinstance(iq_array, np.ndarray) or pd.isna(iq_array).all() or len(iq_array) <= meta_elements:
            return np.array([np.nan])
        actual_iq_data = iq_array[meta_elements:]
        if len(actual_iq_data) == 0 or len(actual_iq_data) % 2 != 0:
            return np.array([np.nan])
        amplitudes = []
        for i in range(0, len(actual_iq_data), 2):
            I_val = actual_iq_data[i]
            Q_val = actual_iq_data[i+1]
            if pd.isna(I_val) or pd.isna(Q_val):
                amplitudes.append(np.nan)
            else:
                amplitude = np.sqrt(float(I_val)**2 + float(Q_val)**2)
                amplitudes.append(amplitude)
        return np.array(amplitudes)

    expected_int_array_len = None
    if not df['csi_raw'].dropna().empty:
        for csi_raw_val in df['csi_raw'].dropna():
            parsed_temp = parse_csi_raw_to_int_array(csi_raw_val)
            if isinstance(parsed_temp, np.ndarray) and len(parsed_temp) > 0 and not np.all(np.isnan(parsed_temp)):
                expected_int_array_len = len(parsed_temp)
                print(f"Inferred raw integer array length from csi_raw (expected_int_array_len): {expected_int_array_len}")
                break
    
    if expected_int_array_len is None or expected_int_array_len == 0:
        print("Could not determine a valid expected raw integer array length from csi_raw. Skipping detailed CSI parsing.")
    else:
        print("Parsing 'csi_raw' to integer arrays...")
        df['csi_parsed_int_list'] = df['csi_raw'].apply(lambda x: parse_csi_raw_to_int_array(x, expected_int_array_len))
        print(f"Calculating CSI amplitudes, skipping first {num_meta_elements} elements as metadata...")
        df['csi_amplitudes_list'] = df['csi_parsed_int_list'].apply(lambda x: calculate_amplitudes_from_iq_pairs(x, meta_elements=num_meta_elements))

        expected_amplitude_len = None
        if not df['csi_amplitudes_list'].dropna().empty:
            for amp_array in df['csi_amplitudes_list'].dropna():
                if isinstance(amp_array, np.ndarray) and len(amp_array) > 0 and not np.all(np.isnan(amp_array)) and not (len(amp_array) == 1 and np.isnan(amp_array[0])):
                    expected_amplitude_len = len(amp_array)
                    print(f"Inferred number of CSI amplitudes (expected_amplitude_len): {expected_amplitude_len}")
                    break
        
        if expected_amplitude_len is None or expected_amplitude_len == 0:
            print("Could not determine a valid expected amplitude length. Skipping detailed CSI amplitude analysis.")
        else:
            df['csi_amplitude_is_valid'] = df['csi_amplitudes_list'].apply(
                lambda x: isinstance(x, np.ndarray) and len(x) == expected_amplitude_len and not np.isnan(x).any()
            )
            print(f"Proportion of valid CSI amplitude entries: {df['csi_amplitude_is_valid'].mean():.2%}")
            
            csi_amp_df_valid_rows = df[df['csi_amplitude_is_valid']].copy()

            if not csi_amp_df_valid_rows.empty:
                csi_amplitude_columns = [f'csi_amp_sub_{i}' for i in range(expected_amplitude_len)]
                parsed_csi_amp_df_for_join = pd.DataFrame(
                    csi_amp_df_valid_rows['csi_amplitudes_list'].tolist(),
                    columns=csi_amplitude_columns,
                    index=csi_amp_df_valid_rows.index # Use original index for joining
                )
                # Join individual subcarrier amplitudes to the main DataFrame
                df = df.join(parsed_csi_amp_df_for_join)

                if not parsed_csi_amp_df_for_join.empty:
                    plt.figure(figsize=(12, 6))
                    plt.plot(parsed_csi_amp_df_for_join.iloc[0]) # Plot first valid row
                    plt.title(f'Example of Parsed CSI Amplitudes (First Valid Sample - {expected_amplitude_len} subcarriers)')
                    plt.xlabel('Subcarrier Index')
                    plt.ylabel('Amplitude')
                    plt.show()

                # Heatmaps (consider sampling for performance if data is very large)
                for scenario_name in df['scenario'].unique(): # Use main df for unique scenarios
                    scenario_csi_data = df[ (df['scenario'] == scenario_name) & (df['csi_amplitude_is_valid']) ][csi_amplitude_columns]
                    if scenario_csi_data.empty:
                        print(f"No valid CSI amplitude data for heatmap in scenario: {scenario_name}.")
                        continue
                    
                    max_samples_heatmap = 500 # Limit samples for heatmap readability
                    if len(scenario_csi_data) > max_samples_heatmap:
                        plot_data_heatmap = scenario_csi_data.sample(n=max_samples_heatmap, random_state=42).sort_index()
                    else:
                        plot_data_heatmap = scenario_csi_data.sort_index()

                    plt.figure(figsize=(15, 7))
                    ytick_step = max(1, expected_amplitude_len // 10)
                    xtick_step = max(1, len(plot_data_heatmap) // 10)
                    sns.heatmap(plot_data_heatmap.T, cmap='viridis', 
                                yticklabels=ytick_step, xticklabels=xtick_step if xtick_step > 0 else False) # Handle xtick_step=0
                    plt.title(f'CSI Amplitude Heatmap for Scenario: {scenario_name} (Max {len(plot_data_heatmap)} samples)')
                    plt.xlabel('Time Sample Index (within scenario, sorted)')
                    plt.ylabel('Subcarrier Index (Amplitude)')
                    plt.show()
                
                # Calculate summary statistics (mean, std, min, max) from individual subcarriers
                df['csi_amp_mean'] = df[csi_amplitude_columns].mean(axis=1)
                df['csi_amp_std']  = df[csi_amplitude_columns].std(axis=1)
                df['csi_amp_min']  = df[csi_amplitude_columns].min(axis=1)
                df['csi_amp_max']  = df[csi_amplitude_columns].max(axis=1)

                csi_amp_stat_features = ['csi_amp_mean', 'csi_amp_std', 'csi_amp_min', 'csi_amp_max']
                for feature in csi_amp_stat_features:
                    if feature in df.columns and df[feature].notna().any():
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        sns.histplot(df[feature].dropna(), kde=True, bins=30)
                        plt.title(f'Distribution of {feature}')
                        plt.subplot(1, 3, 2)
                        sns.boxplot(x='occupancy', y=feature, data=df.dropna(subset=[feature]))
                        plt.title(f'{feature} vs. Occupancy')
                        plt.subplot(1, 3, 3)
                        sns.boxplot(x='scenario', y=feature, data=df.dropna(subset=[feature]))
                        plt.title(f'{feature} vs. Scenario')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.show()

                # --- NEW: Selected Individual Subcarrier Analysis ---
                print("\n--- NEW: Selected Individual Subcarrier Analysis ---")
                # Select a few subcarriers, avoiding known problematic ones if any (e.g., first or null ones)
                # Example: If 64 subcarriers, indices 0-63.
                selected_subcarrier_indices = [5, 15, 30, 45, 60] # Example indices
                selected_subcarrier_cols = [f'csi_amp_sub_{i}' for i in selected_subcarrier_indices if f'csi_amp_sub_{i}' in df.columns]

                if selected_subcarrier_cols:
                    for sub_col in selected_subcarrier_cols:
                        plt.figure(figsize=(12, 6))
                        sns.boxplot(x='scenario', y=sub_col, data=df.dropna(subset=[sub_col]))
                        plt.title(f'Amplitude of {sub_col} across Scenarios')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.show()
                else:
                    print("Selected subcarrier columns not found in DataFrame.")
                # --- END NEW Selected Subcarriers ---

                # --- NEW: CSI Amplitude Difference Analysis (within scenario) ---
                print("\n--- NEW: CSI Amplitude Difference Analysis (within scenario) ---")
                # Calculate diffs for each subcarrier within each scenario
                csi_diff_list = []
                for scenario in df['scenario'].unique():
                    scenario_data = df[df['scenario'] == scenario][csi_amplitude_columns].copy()
                    scenario_diffs = scenario_data.diff() # diffs along time axis for this scenario
                    csi_diff_list.append(scenario_diffs)
                
                if csi_diff_list:
                    csi_diff_df_within_scenario = pd.concat(csi_diff_list).abs() # Combine and take absolute
                    
                    # Add these as new columns to the main df, aligning by index
                    df['mean_abs_csi_diff_per_ts'] = csi_diff_df_within_scenario.mean(axis=1)
                    df['std_abs_csi_diff_per_ts'] = csi_diff_df_within_scenario.std(axis=1)

                    for new_feat in ['mean_abs_csi_diff_per_ts', 'std_abs_csi_diff_per_ts']:
                        if df[new_feat].notna().any():
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(x='scenario', y=new_feat, data=df.dropna(subset=[new_feat]))
                            plt.title(f'{new_feat} vs. Scenario')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            plt.show()

                            plt.figure(figsize=(10, 5))
                            sns.boxplot(x='occupancy', y=new_feat, data=df.dropna(subset=[new_feat]))
                            plt.title(f'{new_feat} vs. Occupancy')
                            plt.tight_layout()
                            plt.show()
                        else:
                            print(f"Feature {new_feat} has no valid data to plot.")
                else:
                    print("Could not compute CSI differences within scenarios.")
                # --- END NEW CSI Differences ---

                # --- NEW: CSI Spectral Flatness Measure (SFM) Analysis ---
                print("\n--- NEW: CSI Spectral Flatness Measure (SFM) Analysis ---")
                def calculate_sfm(amplitude_row):
                    valid_amps = amplitude_row.dropna() # Ensure no NaNs
                    valid_amps_positive = valid_amps[valid_amps > 1e-9] # Ensure positive for gmean
                    if len(valid_amps_positive) < 2: # Need at least a few points
                        return np.nan
                    amean = np.mean(valid_amps_positive)
                    if amean == 0: # Avoid division by zero
                        return np.nan
                    geomean = gmean(valid_amps_positive)
                    return geomean / amean

                # Apply SFM calculation only to rows with valid CSI amplitudes
                # Create a temporary DataFrame of just the amplitude columns for SFM calculation
                df_csi_amps_for_sfm = df.loc[df['csi_amplitude_is_valid'], csi_amplitude_columns]
                if not df_csi_amps_for_sfm.empty:
                    sfm_values = df_csi_amps_for_sfm.apply(calculate_sfm, axis=1)
                    df.loc[sfm_values.index, 'csi_sfm'] = sfm_values # Assign back to main df using original index

                    if 'csi_sfm' in df.columns and df['csi_sfm'].notna().any():
                        plt.figure(figsize=(12, 6))
                        sns.boxplot(x='scenario', y='csi_sfm', data=df.dropna(subset=['csi_sfm']))
                        plt.title('CSI Spectral Flatness Measure vs. Scenario')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.show()

                        plt.figure(figsize=(10, 5))
                        sns.boxplot(x='occupancy', y='csi_sfm', data=df.dropna(subset=['csi_sfm']))
                        plt.title('CSI Spectral Flatness Measure vs. Occupancy')
                        plt.tight_layout()
                        plt.show()
                    else:
                        print("csi_sfm feature has no valid data to plot or was not calculated.")
                else:
                    print("No valid CSI amplitude data available for SFM calculation.")
                # --- END NEW SFM ---
            else:
                print("No CSI amplitude entries with valid, consistent length found after parsing. Cannot perform detailed CSI amplitude analysis.")

print("\n--- EDA Finished ---")
print("Review the plots and statistics to understand your data.")
print("Key things to check:")
print("- Data types and missing values.")
print("- Distributions of target variables ('occupancy', 'scenario').")
print("- Timestamp continuity and gaps between scenario blocks.")
print("- Distributions of numerical features (original and NEWLY DERIVED) and their relationship with targets.")
print("- Constant or low variance numerical features.")
print("- Distributions of categorical features and their relationship with targets.")
print("- Characteristics of parsed CSI data: individual subcarriers, heatmaps, summary stats, and NEWLY DERIVED CSI features (differences, SFM).")
print("This EDA will guide your preprocessing steps (e.g., feature selection, scaling, encoding, sequence creation).")
print("\nACTION REQUIRED: Please review the 'num_meta_elements' variable in Section 6 and set it correctly based on your CSI data format.")