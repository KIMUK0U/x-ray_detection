# U-Net Medical Image Segmentation - 実装課題

本プロジェクトは、医療画像のセマンティックセグメンテーションをU-Netで実装する学習課題です。

## 📁 プロジェクト構成

```
UNET/
├── Data/
│   ├── train/
│   │   ├── images/     # 学習用画像 (.npy)
│   │   └── labels/     # 学習用ラベル (.npy)
│   ├── val/
│   │   ├── images/     # 検証用画像 (.npy)
│   │   └── labels/     # 検証用ラベル (.npy)
│   └── test/
│       ├── images/     # テスト用画像 (.npy)
│       └── labels/     # テスト用ラベル (.npy)
├── checkpoints/        # 学習済みモデルの保存先
├── Criterion.py        # 損失関数 (実装済み)
├── model.py           # U-Netモデル定義 (要実装)
├── main.py            # 学習スクリプト (要実装)
├── test.py            # 評価スクリプト (要実装)
└── data.py            # データ可視化ツール (実装済み)
```

## 🎯 実装課題

### 課題1: モデル定義 (`model.py`)

`model.py` に以下の2つのクラスを実装してください。

#### 1.1 `DoubleConv` クラス

畳み込みブロックを定義するクラスです。

**実装要件:**
- 2つの畳み込み層 (Conv2d → BatchNorm2d → ReLU) のシーケンス
- Residual接続の実装（入出力チャンネル数が同じ場合のみ）
- 最後にReLU活性化関数を適用

**ヒント:**
```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # WRITE ME: residual接続の条件とconv_blockを定義
        # - in_channels == out_channelsの場合、residual接続を有効化
        # - Conv2d(kernel_size=3, padding=1) → BatchNorm2d → ReLU を2回

    def forward(self, x):
        # WRITE ME: forward処理を実装
        # 1. 入力をidentityとして保存
        # 2. conv_blockを通過
        # 3. residual接続が有効な場合、identityを加算
        # 4. 最終的にReLUを適用
        pass
```

#### 1.2 `UNet` クラス

U-Netアーキテクチャの本体を定義します。

**設計指針:**

| 入力サイズ | 推奨段数 | 理由 |
|----------|---------|------|
| 512x512  | 5段階   | 最小解像度: 16x16 |
| 1024x1024| 6段階   | 最小解像度: 16x16 |

**実装要件:**

1. **エンコーダ（Encoder）**
   - 初期畳み込み: `DoubleConv(n_channels, base_channels)`
   - ダウンサンプリング: `MaxPool2d(2)` → `DoubleConv`
   - チャンネル数の増加パターン: 64 → 128 → 256 → 512 → ...

2. **デコーダ（Decoder）**
   - アップサンプリング: `ConvTranspose2d(kernel_size=2, stride=2)`
   - スキップ接続: エンコーダの特徴マップと結合 (`torch.cat`)
   - 結合後のチャンネル数に注意（2倍になる）
   - `DoubleConv`で特徴抽出

3. **出力層**
   - `Conv2d(base_channels, n_classes, kernel_size=1)`

**実装のポイント:**
```python
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        base_channels = 64  # 基本チャンネル数
        
        # WRITE ME: エンコーダ層を定義
        # self.inc = DoubleConv(n_channels, base_channels)
        # self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(...))
        # ...
        
        # WRITE ME: デコーダ層を定義
        # self.up1 = nn.ConvTranspose2d(...)
        # self.conv1 = DoubleConv(...)  # スキップ接続後のチャンネル数に注意
        # ...
        
        # WRITE ME: 出力層
        # self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # WRITE ME: forward処理を実装
        # 1. エンコーダで特徴マップを保存 (x1, x2, x3, ...)
        # 2. デコーダでアップサンプリング + スキップ接続
        #    - アップサンプリング
        #    - torch.cat([encoder特徴マップ, 現在の特徴マップ], dim=1)
        #    - DoubleConvで処理
        # 3. 出力層で最終予測
        
        logits = self.outc(x)
        return logits
```

---

### 課題2: データ拡張 (`main.py`)

`NpyDataset`クラスの`__getitem__`メソッド内に、データ拡張処理を実装してください。

**実装要件:**
- **訓練時のみ**適用（`self.split == 'train'`）
- 画像とラベルに**同じ変換**を適用
- 実装する拡張:
  1. **左右反転** (Horizontal Flip) - 50%の確率
  2. **回転** (Rotation) - ±20度の範囲、70%の確率
  3. **並進移動** (Translation) - 最大200ピクセル、70%の確率

**重要な注意点:**
- 画像には`cv2.INTER_LINEAR`（線形補間）
- ラベルには`cv2.INTER_NEAREST`（最近傍補間）
- チャンネル次元が失われた場合は`np.expand_dims`で復元

**実装例:**
```python
if self.split == 'train':
    (h, w) = img_arr.shape[:2]
    
    # 1. 左右反転
    if np.random.rand() < 0.5:
        # WRITE ME: np.fliplrを使用
        pass

    # 2. 回転
    if np.random.rand() < 0.7:
        # WRITE ME: cv2.getRotationMatrix2D と cv2.warpAffine を使用
        # angle = np.random.uniform(-20, 20)
        pass
    
    # 3. 並進移動
    if np.random.rand() < 0.7:
        # WRITE ME: 移動量を生成し、cv2.warpAffineで適用
        # tx = np.random.uniform(-200, 200)
        # ty = np.random.uniform(-200, 200)
        pass
    
    # チャンネル次元の復元
    if img_arr.ndim == 2:
        img_arr = np.expand_dims(img_arr, axis=-1)
    if lbl_arr.ndim == 2:
        lbl_arr = np.expand_dims(lbl_arr, axis=-1)
```

---

### 課題3: テスト評価 (`test.py`)

`test.py`のデータセットクラスは`main.py`と同じものを使用するため、同様にデータ拡張部分を実装してください。ただし、**テスト時は拡張を適用しない**ことに注意してください（`split='test'`では何もしない）。

---

## 🔧 環境構築

### 必要なライブラリ

```bash
pip install torch torchvision numpy opencv-python matplotlib scikit-learn tqdm
```

### デバイス設定

コードは自動的に利用可能なデバイスを選択します:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU

---

## 🚀 実行方法

### 1. データの準備

`Data/` ディレクトリに以下の構造でデータを配置してください:
```
Data/
├── train/
│   ├── images/*.npy
│   └── labels/*.npy
├── val/
│   ├── images/*.npy
│   └── labels/*.npy
└── test/
    ├── images/*.npy
    └── labels/*.npy
```

### 2. データの確認

```bash
python data.py
```

指定した`.npy`ファイルの可視化ができます。

### 3. 学習の実行

```bash
cd x-ray_detection
python UNET/main.py
```

**ハイパーパラメータ:**
- `BATCH_SIZE`: 2
- `LEARNING_RATE`: 1e-4
- `EPOCHS`: 20
- `IMG_SIZE`: 512

学習済みモデルは `checkpoints/best_model.pth` に保存されます。

### 4. 評価の実行

```bash
cd x-ray_detection
python UNET/test.py
```

**評価指標:**
- AUC (Area Under the Curve)
- Confusion Matrix (Image-wise)
- Precision (適合率)
- Recall (再現率)
- IoU (Intersection over Union)

評価結果とともに、ROC曲線と予測結果の可視化が表示されます。

---

## 📊 損失関数

`Criterion.py` には以下の損失関数が実装されています（実装済み）:

### `DiceLoss`
セグメンテーションタスクで広く使われる損失関数。クラス不均衡に強い。

$$
\text{Dice Loss} = 1 - \frac{2 \cdot |X \cap Y| + \epsilon}{|X|^2 + |Y|^2 + \epsilon}
$$

### `FocalLoss`
難しいサンプルに重点を置く損失関数。誤分類されやすいピクセルの学習を重視。

$$
\text{Focal Loss} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

### `WeightedCombinedLoss`
Dice LossとFocal Lossを組み合わせた損失関数。

```python
criterion = WeightedCombinedLoss(dice_weight=1, focal_weight=0)
```

---

## 🎓 学習のポイント

### U-Netアーキテクチャの理解

1. **エンコーダ（Encoder）**
   - 画像の特徴を抽出しながら解像度を下げる
   - MaxPoolingで空間情報を圧縮
   - チャンネル数を増やして抽象的な特徴を学習

2. **ボトルネック（Bottleneck）**
   - 最も深い層で最も抽象的な特徴を学習
   - 最小の空間解像度、最大のチャンネル数

3. **デコーダ（Decoder）**
   - アップサンプリングで解像度を復元
   - スキップ接続でエンコーダの空間情報を結合
   - 高解像度の予測マップを生成

### データ拡張の重要性

医療画像は取得が困難なため、データ拡張は必須です:
- **回転**: 撮影角度のバリエーション
- **反転**: 左右対称性の学習
- **並進**: 位置のバリエーション

### デバッグのヒント

1. **テンソルサイズの確認**
   ```python
   print(f"Encoder output: {x1.shape}")
   print(f"After upsampling: {x.shape}")
   print(f"After concat: {torch.cat([x1, x], dim=1).shape}")
   ```

2. **少量データでのテスト**
   - 最初は`EPOCHS=1`、`BATCH_SIZE=1`で動作確認
   - データ数を減らして（例: 10サンプル）オーバーフィットを確認

3. **可視化**
   - `data.py`でデータを確認
   - 学習中の損失値をプロット
   - `test.py`で予測結果を可視化

---

## 📝 提出物

以下のファイルを完成させて提出してください:

1. ✅ `model.py` - U-Netモデルの実装
2. ✅ `main.py` - データ拡張部分の実装
3. ✅ `test.py` - データ拡張部分の実装（テスト時は適用しない）
4. 📊 実行結果のスクリーンショット
   - 学習時の損失推移
   - テスト評価の結果（AUC, Precision, Recall等）
   - ROC曲線
   - 予測結果の可視化（5サンプル程度）

---

## 🔍 参考文献

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 💡 トラブルシューティング

### よくあるエラー

1. **RuntimeError: Sizes of tensors must match**
   - スキップ接続時のテンソルサイズ不一致
   - → エンコーダとデコーダの対応を確認

2. **RuntimeError: CUDA out of memory**
   - GPUメモリ不足
   - → `BATCH_SIZE`を減らす（1〜2推奨）

3. **AssertionError: File mismatch**
   - 画像とラベルのファイル数が不一致
   - → `Data/`ディレクトリ構造を確認

4. **次元が合わない**
   - データ拡張後にチャンネル次元が消失
   - → `np.expand_dims(arr, axis=-1)` で復元

---

## 📧 質問・サポート

実装中に不明点があれば、以下を確認してください:
1. エラーメッセージの全文
2. 該当コード箇所
3. テンソルのshape情報
4. データセットの構造

Good luck with your implementation! 🎉
