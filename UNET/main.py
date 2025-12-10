import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    from UNET.Criterion import WeightedCombinedLoss
except ImportError:
    from Criterion import WeightedCombinedLoss
try:
    from UNET.model import UNet
except ImportError:
    from model import UNet

# ==========================================
# 1. 設定 & ハイパーパラメータ
# ==========================================
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

BATCH_SIZE = 2      
LEARNING_RATE = 1e-4
EPOCHS = 20
IMG_SIZE = 512     
NUM_WORKERS = 0     # Mac/Windowsでのエラー回避のため0推奨

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
CHECKPOINT_PATH = "./UNET/checkpoints/best_model.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. データセットクラス
# ==========================================
import cv2 

# ==========================================
# 2. データセットクラス (データ拡張ロジック追加)
# ==========================================
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 
import random # ランダムな移動量を生成するためにインポート

# ==========================================
# 2. データセットクラス (データ拡張ロジック追加)
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
# 4. 評価関数 & 学習ループ
# ==========================================
def dice_coeff(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for imgs, lbls in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Val  "):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            val_dice += dice_coeff(preds, lbls).item()
            
    return val_loss / len(loader), val_dice / len(loader)
    
def main():
    print(f"Using Device: {DEVICE}")
    
    train_dataset = NpyDataset(DATA_DIR, split='train')
    val_dataset = NpyDataset(DATA_DIR, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Train Data: {len(train_dataset)}")
    print(f"Val Data:   {len(val_dataset)}")
    
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    # model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)) # チェックポイントからのロード（必要に応じてコメント解除）
    criterion = WeightedCombinedLoss(dice_weight=1, focal_weight=0) # Dice Lossのみ使用
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print("\nStart Training...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"Best Model Saved! (Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    main()