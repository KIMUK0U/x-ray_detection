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
# Removed sklearn metrics - using proper segmentation metrics instead

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

        # ファイル名も返す (可視化用)
        filename = os.path.basename(img_path)
        return img_tensor, lbl_tensor, filename

# ==========================================
# 3. 評価ロジック (セグメンテーション用の正しい評価指標)
# ==========================================
def dice_score(pred_mask, true_mask, smooth=1e-5):
    """Dice係数 (F1スコアと等価) を計算する"""
    intersection = (pred_mask * true_mask).sum()
    return (2.0 * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)

def jaccard_index(pred_mask, true_mask, smooth=1e-5):
    """IoU (Intersection over Union) を計算する"""
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred_mask, true_mask):
    """ピクセル単位の正解率を計算する"""
    correct = (pred_mask == true_mask).sum()
    total = pred_mask.numel()
    return correct / total

def evaluate_model(model, data_loader, device):
    """セグメンテーションモデルの評価"""
    model.eval()

    # ピクセル単位の評価指標用
    total_dice = 0.0
    total_iou = 0.0
    total_pixel_acc = 0.0

    # ピクセル単位の混合行列用
    pixel_tp = 0  # True Positive (正しく陽性と予測)
    pixel_fp = 0  # False Positive (誤って陽性と予測)
    pixel_fn = 0  # False Negative (誤って陰性と予測)
    pixel_tn = 0  # True Negative (正しく陰性と予測)

    total_images = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs, targets = batch[0], batch[1]  # Unpack first two items (image, label)
            inputs = inputs.to(device)
            targets = targets.to(device)  # [B, 1, H, W]

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds_binary = (probs > 0.5).float()  # [B, 1, H, W]

            for i in range(inputs.size(0)):
                pred_mask = preds_binary[i].squeeze()  # [H, W]
                true_mask = targets[i].squeeze()       # [H, W]

                # セグメンテーション評価指標を計算
                total_dice += dice_score(pred_mask, true_mask).item()
                total_iou += jaccard_index(pred_mask, true_mask).item()
                total_pixel_acc += pixel_accuracy(pred_mask, true_mask).item()

                # ピクセル単位の混合行列を計算
                pred_flat = pred_mask.flatten().cpu().numpy()
                true_flat = true_mask.flatten().cpu().numpy()

                pixel_tp += np.sum((pred_flat == 1) & (true_flat == 1))
                pixel_fp += np.sum((pred_flat == 1) & (true_flat == 0))
                pixel_fn += np.sum((pred_flat == 0) & (true_flat == 1))
                pixel_tn += np.sum((pred_flat == 0) & (true_flat == 0))

                total_images += 1

    # 平均値を計算
    mean_dice = total_dice / total_images
    mean_iou = total_iou / total_images
    mean_pixel_acc = total_pixel_acc / total_images

    # ピクセル単位のPrecision, Recall, F1を計算
    pixel_precision = pixel_tp / (pixel_tp + pixel_fp) if (pixel_tp + pixel_fp) > 0 else 0
    pixel_recall = pixel_tp / (pixel_tp + pixel_fn) if (pixel_tp + pixel_fn) > 0 else 0
    pixel_f1 = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0

    metrics = {
        "Total Images": total_images,
        "Mean Dice Score": mean_dice,
        "Mean IoU (Jaccard)": mean_iou,
        "Mean Pixel Accuracy": mean_pixel_acc,
        "Pixel-wise Precision": pixel_precision,
        "Pixel-wise Recall": pixel_recall,
        "Pixel-wise F1 Score": pixel_f1,
        "Pixel TP": pixel_tp,
        "Pixel FP": pixel_fp,
        "Pixel FN": pixel_fn,
        "Pixel TN": pixel_tn,
    }
    return metrics

# ROC curve plotting removed - not appropriate for segmentation tasks

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
    print("\n🔬 Starting evaluation on Test Data...")
    metrics = evaluate_model(model, test_loader, DEVICE)

    # 結果の表示
    print("\n" + "="*60)
    print("      Segmentation Model Evaluation Metrics")
    print("="*60)
    print(f"Total Images: {metrics['Total Images']}")

    print("\n--- Image-Level Metrics (Average per Image) ---")
    print(f"Mean Dice Score (F1):      {metrics['Mean Dice Score']:.4f}")
    print(f"Mean IoU (Jaccard Index):  {metrics['Mean IoU (Jaccard)']:.4f}")
    print(f"Mean Pixel Accuracy:       {metrics['Mean Pixel Accuracy']:.4f}")

    print("\n--- Pixel-Level Metrics (Across All Pixels) ---")
    print(f"Precision: {metrics['Pixel-wise Precision']:.4f}")
    print(f"Recall:    {metrics['Pixel-wise Recall']:.4f}")
    print(f"F1 Score:  {metrics['Pixel-wise F1 Score']:.4f}")

    print("\n--- Pixel-wise Confusion Matrix ---")
    print(f"True Positives (TP):  {metrics['Pixel TP']:,}")
    print(f"False Positives (FP): {metrics['Pixel FP']:,}")
    print(f"False Negatives (FN): {metrics['Pixel FN']:,}")
    print(f"True Negatives (TN):  {metrics['Pixel TN']:,}")
    print("="*60)
    
    # 実際のデータと予測の視覚化
    print("\n🖼️ Displaying sample predictions...")
    num_to_plot = min(5, len(test_dataset)) 
    plot_predictions(model, test_dataset, DEVICE, num_images=num_to_plot)

if __name__ == "__main__":
    main()
