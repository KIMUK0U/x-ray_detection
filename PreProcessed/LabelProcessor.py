import os
import numpy as np
import cv2
from typing import List

# ==========================================
# 2. ラベル処理クラス (修正版)
# ==========================================
class LabelProcessor:
    """
    ラベルデータの読み込み、パース、整形を行うクラス。
    Faster-RCNNモードで保存済み.txtファイルを読み込めるように修正。
    """
    
    ORIGINAL_SIZE = 2048

    def __init__(self, target_size=(512, 512), mode='unet'):
        self.target_size = target_size
        self.mode = mode

    def _parse_faster_rcnn_file(self, label_path: str) -> List[List[int]]:
        """
        保存された [x_min y_min x_max y_max class_id] 形式の
        .txtファイルを読み込み、リストのリストとして返すヘルパー関数。
        """
        data = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    # スペース区切りでパースし、intに変換
                    parts = list(map(int, line.strip().split()))
                    if len(parts) == 5:
                        data.append(parts)
        except Exception as e:
            # ファイルが見つからない、または形式エラーの場合
            print(f"警告: ラベルファイルの読み込みまたはパースに失敗しました ({label_path}): {e}")
            pass
        return data
    
    def _parse_yolo_file(self, label_path: str) -> List[List[float]]:
        """
        [YOLO用 - ★追加★]
        保存された [class_id x_center y_center width height] 形式 (0-1正規化済み) の
        .txtファイルを読み込み、リストのリストとして返すヘルパー関数。
        """
        data = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    # スペース区切りでパースし、floatに変換 (class_idも一旦floatになる)
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        data.append(parts)
        except Exception as e:
            print(f"警告: YOLOラベルの読み込み失敗 ({label_path}): {e}")
            pass
        return data
    
    def process(self, label_path: str, has_nodule: bool = True):
        """
        ラベルデータを処理し、指定されたモデル形式に整形して返します。

        label_path: ラベルファイルのパス (str)
        has_nodule: 結節があるかどうか
        """
        
        # -------------------------------------
        # --- U-Net用 (セグメンテーションマスク) ---
        # -------------------------------------
        if self.mode == 'unet':
            # ★ U-Netのロジックはマスク処理として正しく、そのまま維持 ★
            if has_nodule and os.path.exists(label_path):
                # マスク読み込み (グレースケール)
                mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                     mask = np.zeros(self.target_size, dtype=np.uint8)
                else:
                    # 最近傍補間 ('INTER_NEAREST') でリサイズするのが正しい
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            else:
                # 結節なし、またはファイルがない場合は真っ黒なマスク
                mask = np.zeros(self.target_size, dtype=np.uint8)
            
            # 0-1のバイナリマスクに変換し、チャネル軸を追加
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1) # (H, W, 1)
            return mask

        # -------------------------------------
        # --- YOLO用 (Bounding Box) ---
        # -------------------------------------
        elif self.mode == 'yolo':
            # ★ 修正点: ファイルパスからラベルを読み込み、パースする ★
            if has_nodule and os.path.exists(label_path):
                # 1. ファイルを読み込み、正規化された座標のリストを取得
                labels_list = self._parse_yolo_file(label_path)
            else:
                labels_list = []
            
            # 2. NumPy配列に変換して返す (YOLOラベルは既に正規化されているため、リスケールは不要)
            # shape: (N, 5), dtype: float32
            if not labels_list:
                return np.zeros((0, 5), dtype=np.float32)

            # class_idをintとして扱うため、一旦floatで読み込んだ後、必要に応じてキャスト
            return np.array(labels_list, dtype=np.float32)

        # --- Faster R-CNN用 (Bounding Box) ---
        elif self.mode == 'faster_rcnn':
            # ★ 修正点: ファイルパスからラベルを読み込み、パースする ★
            if has_nodule and os.path.exists(label_path):
                # 1. ファイルを読み込み、絶対座標のリストを取得
                labels_list = self._parse_faster_rcnn_file(label_path)
            else:
                labels_list = []
            
            labels_2048 = np.array(labels_list, dtype=np.float32)

            if labels_2048.size == 0:
                # 結節がない場合、空の配列を返す
                return np.zeros((0, 5), dtype=np.float32)

            # 2. ターゲットサイズ (例: 512) に座標をリスケール (正規化)
            scale_x = self.target_size[0] / self.ORIGINAL_SIZE
            scale_y = self.target_size[1] / self.ORIGINAL_SIZE

            # 3. 座標をリスケールして返す
            scaled_labels = np.copy(labels_2048)
            scaled_labels[:, [0, 2]] *= scale_x
            scaled_labels[:, [1, 3]] *= scale_y
            
            return scaled_labels
        
        return None