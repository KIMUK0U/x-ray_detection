import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 設定
# ==========================================
NPY_FILE_PATH = 'UNET/Data/test/labels/00023.npy'

# ==========================================
# 可視化関数
# ==========================================
def visualize_npy_mask(file_path: str):
    """NPYファイルからマスクデータを読み込み、可視化する"""
    if not os.path.exists(file_path):
        print(f"❌ エラー: ファイルが見つかりません: {file_path}")
        return

    try:
        # 1. NPYファイルの読み込み
        mask_data = np.load(file_path)
        
        # 2. 形状の確認と調整
        # データは (H, W, 1) または (1, H, W) 形式であることが多いため、
        # 描画のために (H, W) に調整します (チャンネル次元を削除)。
        if mask_data.ndim == 3:
            # (H, W, 1) -> (H, W) または (1, H, W) -> (H, W)
            mask_data = np.squeeze(mask_data)
        
        # 3. データ型の確認 (バイナリマスクであることを前提)
        # float型であることが多いため、描画時はそのまま使用します。

        print(f"✅ ファイル名: {os.path.basename(file_path)}")
        print(f"✅ データの形状: {mask_data.shape}")
        print(f"✅ データ型: {mask_data.dtype}")
        
        # 4. 可視化 (Matplotlibを使用)
        plt.figure(figsize=(6, 6))
        # cmap='gray' で白黒画像として表示
        # interpolation='nearest' でピクセルを鮮明に表示
        plt.imshow(mask_data, cmap='gray', interpolation='nearest') 
        plt.title(f"Visualized Mask from {os.path.basename(file_path)}")
        plt.colorbar(label='Pixel Value (0=Background, 1=Foreground)')
        plt.axis('off') # 軸を非表示
        plt.show()

    except Exception as e:
        print(f"❌ エラー: NPYファイルの読み込みまたは処理に失敗しました: {e}")

# ==========================================
# 実行
# ==========================================
if __name__ == "__main__":
    visualize_npy_mask(NPY_FILE_PATH)