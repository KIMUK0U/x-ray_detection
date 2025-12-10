import pandas as pd
import numpy as np
import os
from typing import Dict, List

# --- FasterRCNNLerLabelGenerator クラス定義 ---
class FasterRCNNLerLabelGenerator:
    """
    JSRTデータセットの臨床データからFaster R-CNN用のラベルを作成するクラス。
    """
    
    CLASS_MAPPING = {'benign': 1, 'malignant': 2}
    ORIGINAL_SIZE = 2048

    def __init__(self, csv_path: str, pixel_spacing: float = 0.175):
        self.pixel_spacing = pixel_spacing
        
        try:
            # 【ローカル実行用に修正】渡された csv_path をそのまま使用してファイルを読み込む
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # ローカルでファイルが見つからない場合に適切なエラーを出す
            raise FileNotFoundError(f"エラー: CSVファイルが指定されたパスに見つかりません -> {csv_path}")

        # 以下は前処理ロジック
        self.df = self.df.dropna(subset=['benign/ malignant']).copy()
        
        self.df['Size[mm]'] = pd.to_numeric(self.df['Size[mm]'], errors='coerce')
        self.df['X-cor'] = pd.to_numeric(self.df['X-cor'], errors='coerce')
        self.df['Y-cor'] = pd.to_numeric(self.df['Y-cor'], errors='coerce')
        self.df = self.df.dropna(subset=['Size[mm]', 'X-cor', 'Y-cor'])

    def _mm_to_pixel(self, mm_size):
        return mm_size / self.pixel_spacing

    def generate_labels(self):
        self.df['Size_px'] = self.df['Size[mm]'].apply(self._mm_to_pixel)
        self.df['class_id'] = self.df['benign/ malignant'].map(self.CLASS_MAPPING)

        half_size_px = self.df['Size_px'] / 2
        
        self.df['x_min'] = self.df['X-cor'] - half_size_px
        self.df['y_min'] = self.df['Y-cor'] - half_size_px
        self.df['x_max'] = self.df['X-cor'] + half_size_px
        self.df['y_max'] = self.df['Y-cor'] + half_size_px
        
        for col in ['x_min', 'y_min', 'x_max', 'y_max']:
            self.df[col] = np.rint(self.df[col]).clip(0, self.ORIGINAL_SIZE - 1).astype(int)
        
        output_cols = ['ImageFileName', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
        result_df = self.df[output_cols]
        
        grouped_labels = result_df.groupby('ImageFileName').apply(
            lambda x: x[['x_min', 'y_min', 'x_max', 'y_max', 'class_id']].values.tolist()
        ).to_dict()
        
        return grouped_labels

# --- ラベル保存関数 ---

def save_faster_rcnn_labels(labels_dict: Dict[str, List[List[int]]], output_dir: str):
    """
    Faster R-CNNのラベル辞書を、画像ごとにテキストファイルとして保存する。
    ファイル形式: [x_min y_min x_max y_max class_id] (スペース区切り)
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for filename, labels in labels_dict.items():
        # ファイル名を決定 (例: JPCLN001.dcm -> JPCLN001.txt)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(output_path, 'w') as f:
            for label in labels:
                # [x_min, y_min, x_max, y_max, class_id] をスペース区切りで書き出す
                line = " ".join(map(str, label))
                f.write(line + "\n")
                
        count += 1
        
    print(f"\n✅ {count}個のラベルファイルを {output_dir} に保存しました。")
    return output_dir

# ==========================================
# 実行ブロック
# ==========================================
if __name__ == "__main__":
    # ファイル名 (ローカルでの正しいパス)
    FILE_NAME_CSV = "./Data/JSRT_DB_ClinicalData_0613_2018_2.csv"
    PIXEL_SPACING = 0.175  
    # 出力先ディレクトリ
    OUTPUT_DIR = "./Data/nodule/Labels/F-RCNN"

    # 1. ラベルの生成
    # ローカルでは FILE_NAME_CSV が使用されます
    label_generator = FasterRCNNLerLabelGenerator(FILE_NAME_CSV, pixel_spacing=PIXEL_SPACING)
    faster_rcnn_labels_output = label_generator.generate_labels()

    # 2. 結果の一部表示
    print("--- Faster R-CNN Label Generation Successful ---")
    print("最初の3つの画像の生成されたラベル:")
    for i, (filename, labels) in enumerate(faster_rcnn_labels_output.items()):
        if i >= 3:
            break
        print(f"  ファイル名: {filename}")
        print(f"  ラベル: {labels}")
        print("-" * 30)

    # 3. ラベルの保存
    # Data/Labelsフォルダとそのサブフォルダ F_RCNNLabel を作成し、ファイルを保存します。
    saved_dir = save_faster_rcnn_labels(faster_rcnn_labels_output, OUTPUT_DIR)