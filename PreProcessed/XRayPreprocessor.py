import numpy as np
import pydicom
import cv2
from sklearn.model_selection import train_test_split
# ==========================================
# 1. 画像前処理クラス (カスタマイズ推奨)
# ==========================================
class XRayPreprocessor:
    """
    DICOM画像の読み込み、正規化、強調処理などを行うクラス。
    実験段階でここの処理内容を変更してください。
    """
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def read_dicom(self, path):
        """DICOMを読み込み、ピクセル配列(numpy)として返す"""
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            
            # モノクロ反転の処理 (Photometric InterpretationがMONOCHROME1の場合など)
            if hasattr(dcm, "PhotometricInterpretation"):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    img = np.max(img) - img
            return img
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None

    def normalize(self, img):
        """0-1の範囲に正規化 (Min-Max Scaling)"""
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min == 0:
            return img
        return (img - img_min) / (img_max - img_min)

    def apply_clahe(self, img):
        """コントラスト制限付き適応ヒストグラム均等化 (CLAHE)"""
        # 8bit (0-255) に変換してから適用するのが一般的
        img_uint8 = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_uint8)
        return img_eq.astype(np.float32) / 255.0  # 再び0-1に戻す

    def apply_unsharp_mask(self, img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """アンシャープマスクによる鮮鋭化"""
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharpened = float(amount + 1) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, np.ones(sharpened.shape))
        return sharpened

    def resize(self, img):
        """指定サイズへのリサイズ"""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

    def run(self, dicom_path):
        """
        前処理パイプラインの実行
        ここでの処理順序や有効/無効を調整してください。
        """
        # 1. 読み込み
        img = self.read_dicom(dicom_path)
        if img is None: return None

        # 2. 正規化 (0-1)
        img = self.normalize(img)

        # 3. リサイズ
        img = self.resize(img)

        # 4. コントラスト強調 (CLAHE) - 必要に応じてコメントアウト
        img = self.apply_clahe(img)

        # 5. アンシャープマスク - 必要に応じてコメントアウト
        # img = self.apply_unsharp_mask(img)

        # 6. チャンネル次元の追加 (H, W) -> (C, H, W) or (H, W, C)
        # PyTorchなどでは (C, H, W) が一般的ですが、ここでは汎用的な (H, W, 1) にします
        img = np.expand_dims(img, axis=-1) 

        return img