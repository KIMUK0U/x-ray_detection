import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, jaccard_score

# モデルインポートの柔軟な記述を使用
try:
    # 訓練時のモデル定義が 'UNET/model.py' にあると仮定
    from UNET.model import UNet
except ImportError:
    # または、現在の実行ディレクトリにある 'model.py' にあると仮定
    from model import UNet

# ==============================================================================
# 0. パスとハイパーパラメータの設定
# ==============================================================================
# 📝 必要に応じてパスを修正してください
CHECKPOINT_PATH = './UNET/checkpoints/best_model.pth' 
DATA_DIR = './UNET/Data'
BATCH_SIZE = 2
# 利用可能な場合はGPUを使用
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# ==========================================
# 2. データセットクラス (学習コードからそのまま流用)
# ==========================================
class NpyDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.lbl_dir = os.path.join(root_dir, split, 'labels')
        self.split = split
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.npy')))
        self.lbl_files = sorted(glob.glob(os.path.join(self.lbl_dir, '*.npy')))
        
        assert len(self.img_files) == len(self.lbl_files), \
            f"File mismatch: {len(self.img_files)} vs {len(self.lbl_files)}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        lbl_path = self.lbl_files[idx]
        
        img_arr = np.load(img_path).astype(np.float32) # (H, W, C)
        lbl_arr = np.load(lbl_path).astype(np.float32) # (H, W, C)

        # ---------------------------------------------
        # データ拡張の適用 (train時のみ)
        # 
        # 実装のポイント:
        # - 画像とラベルに同じ変換を適用する
        # - 画像にはINTER_LINEAR、ラベルにはINTER_NEARESTを使用
        # - 次元が失われた場合はexpand_dimsで復元
        # ---------------------------------------------
        if self.split == 'train':
            # WRITE ME: データ拡張を実装
            pass

        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        lbl_tensor = torch.from_numpy(lbl_arr).permute(2, 0, 1)
        
        return img_tensor, lbl_tensor

# ==========================================
# 3. 評価ロジック (修正版)
# ==========================================
def jaccard_index(pred_mask, true_mask, smooth=1e-5):
    """画像1枚ごとのIoUを計算する"""
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    
    # 🚨 症例単位のROC/AUC計算用のリスト (修正箇所) 🚨
    all_scores_img = []   # 画像ごとのスコア (最大予測確率を使用)
    all_targets_img = []  # 画像ごとのターゲット (異常の有無)
    
    # 症例単位の混合行列カウント用
    true_positives_img = 0
    false_positives_img = 0
    false_negatives_img = 0
    true_negatives_img = 0
    total_images = 0

    with torch.no_grad():
        for inputs, targets, _ in tqdm(data_loader, desc="Eval "):
            inputs = inputs.to(device)
            targets = targets.to(device) # [B, 1, H, W]

            outputs = model(inputs)
            probs = torch.sigmoid(outputs) 
            preds_binary = (probs > 0.5).float() # [B, 1, H, W]
            

            for i in range(inputs.size(0)):
                pred_mask = preds_binary[i].squeeze()
                true_mask = targets[i].squeeze()
                prob_map = probs[i].squeeze() # 確率マップ

                # --- 1. ROC/AUC用のスコアとターゲットを抽出 (症例単位) ---
                has_positive_target = true_mask.sum() > 0 # 画像に異常があるか (ターゲット)
                
                # スコアとして、画像内の最大予測確率を使用
                # 異常の存在に対する確信度が高いほどスコアが高くなる
                score_img = prob_map.max().item() 
                
                all_scores_img.append(score_img)
                all_targets_img.append(1 if has_positive_target else 0)

                # --- 2. IoU基準の混合行列計算 (変更なし) ---
                iou = jaccard_index(pred_mask, true_mask)
                has_positive_pred = pred_mask.sum() > 0

                if iou >= iou_threshold:
                    if has_positive_target:
                        true_positives_img += 1 
                    else:
                        true_negatives_img += 1 
                else:
                    if has_positive_target:
                        false_negatives_img += 1 
                    elif has_positive_pred:
                        false_positives_img += 1 
                    # else: 両方なし＆IoU低はスキップ
                
                total_images += 1

    # --- 評価指標の計算 ---
    
    # IoU基準の混合行列 (Image-wise)
    # ... (混合行列の計算は変更なし) ...
    tp, fp, fn, tn = true_positives_img, false_positives_img, false_negatives_img, true_negatives_img
    precision_img = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_img = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 🚨 ROC/AUC (症例単位のスコアとターゲットを使用) 🚨
    targets_np_img = np.array(all_targets_img)
    scores_np_img = np.array(all_scores_img)
    
    # ターゲットが単一クラスのみの場合の例外処理
    if len(np.unique(targets_np_img)) > 1:
        fpr, tpr, _ = roc_curve(targets_np_img, scores_np_img)
        roc_auc = auc(fpr, tpr)
    else:
        # 全症例が同じクラスの場合
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.5
    
    metrics = {
        "IoU (Pixel Avg, Reference)": 0.0, # ピクセルIoUは不要のため省略または0
        "CM (Image-wise)": np.array([[tn, fp], [fn, tp]]),
        "TN_img": tn, "FP_img": fp, "FN_img": fn, "TP_img": tp,
        "Precision (Image)": precision_img,
        "Recall (Image)": recall_img,
        "Total Images": total_images,
        "ROC Curve (FPR, TPR)": (fpr, tpr),
        "AUC (Image-wise)": roc_auc # 項目名を変更
    }
    return metrics

# --- ROC曲線プロット ---
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right"); plt.grid(True); plt.show()

# --- ラベルと予測の並列画像プロット ---
def plot_predictions(model, dataset, device, num_images=5):
    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))
    
    for i in range(num_images):
        img_tensor, mask_tensor, filename = dataset[i]
        input_img = img_tensor.unsqueeze(0).to(device) # バッチ次元を追加
        
        with torch.no_grad():
            output_logit = model(input_img)
            # 確率に変換し、CPUに戻してnumpyに [H, W]
            pred_prob = torch.sigmoid(output_logit).squeeze().cpu().numpy()
        
        # 画像表示用のnumpy配列 [H, W, C]
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        true_mask_np = mask_tensor.squeeze().cpu().numpy() # [H, W]
        pred_binary_np = (pred_prob > 0.5).astype(np.float32) # [H, W]

        # 1チャンネル画像をカラーで表示するため、必要であればsqueeze()
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
        # 1列目: 元画像
        axes[i, 0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        axes[i, 0].set_title(f'Original Image\n({filename})')
        axes[i, 0].axis('off')

        # 2列目: 正解ラベル
        axes[i, 1].imshow(true_mask_np, cmap='gray')
        axes[i, 1].set_title('True Label (Mask)')
        axes[i, 1].axis('off')

        # 3列目: 予測マスク
        axes[i, 2].imshow(pred_binary_np, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. メイン実行ブロック
# ==========================================
def main():
    print(f"Using Device: {DEVICE}")

    # Testデータセットのロード (split='test'を指定)
    test_dataset = NpyDataset(DATA_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Using Device: {DEVICE}")

    test_dataset = NpyDataset(DATA_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Test Data: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("❌ Error: Test Data is empty. Check your DATA_DIR path and file structure.")
        print(f"Expected files in: {os.path.join(DATA_DIR, 'test', 'images')} and labels.")
        return

    # モデルのインスタンス化とロード (省略)
    try:
        model = UNet(n_channels=1, n_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Model loaded from: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    # 評価の実行
    print("\n🔬 Starting evaluation on Test Data (Image-wise CM, IoU Thresh=0.5)...")
    metrics = evaluate_model(model, test_loader, DEVICE)
    
    # 結果の表示
    print("\n" + "="*50)
    print("      Segmentation Model Evaluation Metrics")
    print("          (IoU Threshold: 0.5)")
    print("="*50)
    print(f"Total Images: {metrics['Total Images']}")
    print(f"AUC (Image-wise, Max Prob): {metrics['AUC (Image-wise)']:.4f}") # 項目名を修正
    
    print("\n--- Confusion Matrix (Image-wise, by IoU) ---")
    print(f"True Positives (TP): {metrics['TP_img']}")
    print(f"False Positives (FP): {metrics['FP_img']}")
    print(f"False Negatives (FN): {metrics['FN_img']}")
    print(f"True Negatives (TN): {metrics['TN_img']}")
    print("---------------------------------------------")
    print(f"Precision (適合率): {metrics['Precision (Image)']:.4f}")
    print(f"Recall (再現率): {metrics['Recall (Image)']:.4f}")

    # ROC曲線のプロット
    plot_roc_curve(metrics["ROC Curve (FPR, TPR)"][0], metrics["ROC Curve (FPR, TPR)"][1], metrics['AUC (Image-wise)'])
    
    # 実際のデータと予測の視覚化
    print("\n🖼️ Displaying sample predictions...")
    num_to_plot = min(5, len(test_dataset)) 
    plot_predictions(model, test_dataset, DEVICE, num_images=num_to_plot)

if __name__ == "__main__":
    main()
