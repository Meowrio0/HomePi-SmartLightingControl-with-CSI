#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_csi_files.py
把指定的 6 個 CSI CSV 檔合併成一個，保留欄位順序
"""

import pandas as pd
import pathlib

# ====== 路徑設定 =============================================================
FOLDER = pathlib.Path("/Users/linjianxun/Desktop/csi_presence_detection_project/datasets/raw_csi_data")          # ← 資料夾位置
OUT_FILE = FOLDER / "csi_merged.csv"                        # 輸出檔
# ============================================================================

# 🔸【1】要合併的檔案清單（只改這裡即可增刪檔案）
file_names = [
    "empty_wifi_bt_off.csv",
    "empty_wifi_bt_on.csv",
    "presence_moving_wifi_bt_on.csv",
    "presence_moving_wifi_bt_off.csv",
    "presence_static_wifi_bt_on.csv",
    "presence_static_wifi_bt_off.csv",
]

# 🔸【2】將檔名轉成完整路徑物件
csv_list = [FOLDER / name for name in file_names]           # ⬅ 新增

def main():
    # 🔸【3】檢查檔案是否都存在
    missing = [p for p in csv_list if not p.exists()]
    if missing:
        print("[ERROR] 找不到下列檔案：")
        for p in missing:
            print("   -", p.name)
        return

    dfs = []
    for fp in csv_list:
        # 🔸【4】避免把之前合併出的檔再讀回來
        if fp.name == OUT_FILE.name:
            continue
        print(f"[INFO] 讀取 {fp.name}")
        df = pd.read_csv(fp)
        dfs.append(df)

    # 🔸【5】縱向合併並重新編排索引
    merged = pd.concat(dfs, ignore_index=True)

    merged.to_csv(OUT_FILE, index=False)
    print(f"[INFO] 合併完成，共 {len(merged)} 列 → {OUT_FILE}")

if __name__ == "__main__":
    main()