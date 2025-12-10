import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

# --- 1. YOLO ラベル生成クラス ---

class YOLOLabelGenerator:
    """
    JSRTデータからYOLO形式の正規化されたラベルを生成するクラス。
    """
    
    # クラスIDのマッピング (YOLOは通常0から始めるが、ここでは良性を0、悪性を1とする)
    CLASS_MAPPING = {'benign': 0, 'malignant': 1}
    
    # 元画像のサイズ (X-cor, Y-corの基準)
    ORIGINAL_SIZE = 2048

    def __init__(self, csv_path: str, pixel_spacing: float = 0.175):
        """
        CSVファイルを読み込み、パラメータを設定する。
        """
        self.pixel_spacing = pixel_spacing
        
        try:
            # 【ローカル実行用に修正】渡された csv_path をそのまま使用してファイルを読み込む
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # ローカルでファイルが見つからない場合に適切なエラーを出す
            raise FileNotFoundError(f"エラー: CSVファイルが指定されたパスに見つかりません -> {csv_path}")

        # 必要な列の前処理 (結節データが存在する行のみ抽出)
        self.df = self.df.dropna(subset=['benign/ malignant']).copy()
        
        self.df['Size[mm]'] = pd.to_numeric(self.df['Size[mm]'], errors='coerce')
        self.df['X-cor'] = pd.to_numeric(self.df['X-cor'], errors='coerce')
        self.df['Y-cor'] = pd.to_numeric(self.df['Y-cor'], errors='coerce')
        self.df = self.df.dropna(subset=['Size[mm]', 'X-cor', 'Y-cor'])

    def _mm_to_pixel(self, mm_size: float) -> float:
        """ミリメートル単位のサイズをピクセル単位に変換する。"""
        return mm_size / self.pixel_spacing

    def generate_labels(self) -> Dict[str, List[List[float]]]:
        """
        DataFrameからYOLO形式の正規化されたラベルを生成し、辞書として返す。
        """
        
        # 1. サイズをピクセル単位に変換
        self.df['Size_px'] = self.df['Size[mm]'].apply(self._mm_to_pixel)
        
        # 2. クラスIDに変換
        self.df['class_id'] = self.df['benign/ malignant'].map(self.CLASS_MAPPING)

        # 3. 正規化された中心座標の計算
        self.df['x_center_norm'] = self.df['X-cor'] / self.ORIGINAL_SIZE
        self.df['y_center_norm'] = self.df['Y-cor'] / self.ORIGINAL_SIZE
        
        # 4. 正規化された幅と高さの計算 (w = h)
        self.df['w_norm'] = self.df['Size_px'] / self.ORIGINAL_SIZE
        self.df['h_norm'] = self.df['Size_px'] / self.ORIGINAL_SIZE
        
        # 5. 正規化された座標のクリッピング (0.0から1.0の範囲に収める)
        # 中心座標はクリッピングせず、幅/高さをクリッピングすることが多いが、ここでは簡略化のため座標全体をクリッピング
        for col in ['x_center_norm', 'y_center_norm', 'w_norm', 'h_norm']:
            self.df[col] = self.df[col].clip(0.0, 1.0)
        
        # 6. 出力形式に整形: [class_id, x_center_norm, y_center_norm, w_norm, h_norm]
        output_cols = ['ImageFileName', 'class_id', 'x_center_norm', 'y_center_norm', 'w_norm', 'h_norm']
        result_df = self.df[output_cols]
        
        # 7. 画像ファイル名ごとにグループ化し、ラベルのリストを作成して辞書として返す
        grouped_labels = result_df.groupby('ImageFileName').apply(
            lambda x: x[['class_id', 'x_center_norm', 'y_center_norm', 'w_norm', 'h_norm']].values.tolist()
        ).to_dict()
        
        return grouped_labels

# --- 2. ラベル保存関数 ---

def save_yolo_labels(labels_dict: Dict[str, List[List[float]]], output_dir: str):
    """
    YOLOのラベル辞書を、画像ごとにテキストファイルとして保存する。
    ファイル形式: [class_id x_center_norm y_center_norm w_norm h_norm] (スペース区切り)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for filename, labels in labels_dict.items():
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(output_path, 'w') as f:
            for label in labels:
                # 浮動小数点数（4桁）をスペース区切りで書き出す
                # class_idは整数に変換
                line = f"{int(label[0])} {label[1]:.4f} {label[2]:.4f} {label[3]:.4f} {label[4]:.4f}"
                f.write(line + "\n")
                
        count += 1
        
    print(f"\n✅ {count}個のYOLOラベルファイルを {output_dir} に保存しました。")
    return output_dir

# --- 3. 実行ブロック ---

if __name__ == "__main__":
    
    # ファイル名 (アップロードされた結節データファイルを使用)
    FILE_NAME_CSV = "./Data/JSRT_DB_ClinicalData_0613_2018_2.csv"
    PIXEL_SPACING = 0.175  
    OUTPUT_DIR = "./Data/nodule/Labels/YOLO"

    # 1. ラベルの生成
    label_generator = YOLOLabelGenerator(FILE_NAME_CSV, pixel_spacing=PIXEL_SPACING)
    yolo_labels_output = label_generator.generate_labels()

    # 2. 結果の一部表示
    print("--- YOLO Label Generation Successful ---")
    print("最初の3つの画像の生成されたラベル:")
    for i, (filename, labels) in enumerate(yolo_labels_output.items()):
        if i >= 3:
            break
        print(f"  ファイル名: {filename}")
        print(f"  ラベル: {labels[0]}")
        print("-" * 30)

    # 3. ラベルの保存
    saved_dir = save_yolo_labels(yolo_labels_output, OUTPUT_DIR)

    # 4. 保存されたファイルの一部を確認
    print(f"\n保存ディレクトリ: {saved_dir}")
    if os.path.exists(saved_dir):
        sample_files = sorted(os.listdir(saved_dir))[:5]
        print(f"保存されたファイル名 (一部): {sample_files}")
        
        # 最初のファイルを読み込んで中身を確認
        if sample_files:
            sample_path = os.path.join(saved_dir, sample_files[0])
            print(f"\nサンプルファイルの中身 ({sample_files[0]}):")
            with open(sample_path, 'r') as f:
                print(f.read())
                