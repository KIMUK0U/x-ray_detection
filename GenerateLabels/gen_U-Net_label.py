import pandas as pd
import numpy as np
import os
import cv2
from typing import Dict, List, Any

class UNetLabelGenerator:
    """
    JSRTデータからU-Net用のセグメンテーションマスク（円形）を生成するクラス。
    """
    
    ORIGINAL_SIZE = 2048
    MASK_VALUE = 255 # 結節領域の値

    def __init__(self, csv_path: str, pixel_spacing: float = 0.175):
        """
        CSVファイルを読み込み、パラメータを設定する。
        （ローカルでの実行時には、csv_pathにご自身のファイルパスを渡してください）
        """
        self.pixel_spacing = pixel_spacing
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"ファイル {csv_path} が見つかりません。")

        # 結節データが存在する行のみを抽出・前処理
        self.df = self.df.dropna(subset=['benign/ malignant']).copy()
        self.df['Size[mm]'] = pd.to_numeric(self.df['Size[mm]'], errors='coerce')
        self.df['X-cor'] = pd.to_numeric(self.df['X-cor'], errors='coerce')
        self.df['Y-cor'] = pd.to_numeric(self.df['Y-cor'], errors='coerce')
        self.df = self.df.dropna(subset=['Size[mm]', 'X-cor', 'Y-cor'])

    def _mm_to_radius_pixel(self, mm_size: float) -> int:
        """ミリメートル単位の直径をピクセル単位の半径に変換する。"""
        # 半径 (mm) / ピクセル間隔 (mm/pixel)
        return int(np.round((mm_size / 2) / self.pixel_spacing))

    def generate_masks(self) -> Dict[str, np.ndarray]:
        """
        画像ファイル名ごとに、結節領域が円形で描画されたマスク画像を生成する。
        """
        masks_dict = {}
        
        self.df['Radius_px'] = self.df['Size[mm]'].apply(self._mm_to_radius_pixel)
        grouped = self.df.groupby('ImageFileName')
        
        for filename, group in grouped:
            # 2048x2048 の真っ黒なマスクを初期化
            mask = np.zeros((self.ORIGINAL_SIZE, self.ORIGINAL_SIZE), dtype=np.uint8)
            
            for index, row in group.iterrows():
                center_x = int(np.round(row['X-cor']))
                center_y = int(np.round(row['Y-cor']))
                radius = row['Radius_px']
                
                # 円を描画 (厚さ -1 で塗りつぶし)
                cv2.circle(
                    img=mask, 
                    center=(center_x, center_y), 
                    radius=radius, 
                    color=int(self.MASK_VALUE), 
                    thickness=-1
                )
            
            masks_dict[filename] = mask
        
        return masks_dict

# --- 2. マスク保存関数 ---

def save_unet_masks(masks_dict: Dict[str, np.ndarray], output_dir: str):
    """
    生成されたマスク画像をPNGファイルとして保存する。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for filename, mask_array in masks_dict.items():
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        # cv2.imwriteを使って画像を保存
        cv2.imwrite(output_path, mask_array)
        count += 1
        
    print(f"\n✅ {count}個のU-Netセグメンテーションマスクを {output_dir} に保存しました。")
    return output_dir

# --- 3. 実行ブロック ---

if __name__ == "__main__":
    
    FILE_NAME_CSV = "./Data/JSRT_DB_ClinicalData_0613_2018_2.csv"
    PIXEL_SPACING = 0.175  
    OUTPUT_DIR = "./Data/nodule/Labels/U-Net"

    # 1. マスクの生成
    mask_generator = UNetLabelGenerator(FILE_NAME_CSV, pixel_spacing=PIXEL_SPACING)
    unet_masks_output = mask_generator.generate_masks()

    # 2. マスクの保存
    saved_dir = save_unet_masks(unet_masks_output, OUTPUT_DIR)

    print(f"\n保存先ディレクトリ: {saved_dir}")