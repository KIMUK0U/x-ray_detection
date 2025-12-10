# Medical Image Dataset Builder for Lung Nodule Detection

肺結節検出のための医療画像データセット構築ツール。DICOM形式の胸部X線画像から、U-Net、YOLO、Faster R-CNNの各モデル用にデータセットを自動生成します。

## 📁 プロジェクト構造

```
project_root/
├── Data/
│   ├── nodule/
│   │   ├── Dicom/              # 結節ありDICOM画像
│   │   │   └── *.dcm
│   │   └── Labels/
│   │       ├── U-Net/          # セグメンテーションマスク
│   │       │   └── *_mask.png
│   │       ├── YOLO/           # YOLO形式アノテーション
│   │       │   └── *.txt
│   │       └── F-RCNN/         # Faster R-CNN形式アノテーション
│   │           └── *.txt
│   └── non_nodule/
│       └── **/                 # 結節なしDICOM画像（サブディレクトリ含む）
│           └── *.dcm
│
├── PreProcessed/
│   ├── DataBuilder.py          # メインスクリプト
│   ├── LabelProcessor.py       # ラベル処理
│   └── XRayPreprocessor.py     # 画像前処理
│
└── [OUTPUT]/
    ├── UNET/Data/
    ├── YOLO/Data/
    └── F-RCNN/Data/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

## 🚀 使い方

### 基本的な実行方法

```bash
python PreProcessed/DataBuilder.py
```

### モード設定

`DataBuilder.py`内の`LABEL_MODE`を変更することで、出力形式を選択できます：

```python
LABEL_MODE = 'unet'        # U-Net用（セグメンテーションマスク）
LABEL_MODE = 'yolo'        # YOLO用（正規化されたバウンディングボックス）
LABEL_MODE = 'faster_rcnn' # Faster R-CNN用（絶対座標バウンディングボックス）
```

### カスタマイズ可能なパラメータ

```python
INPUT_ROOT = "./Data"           # 入力データのルートディレクトリ
OUTPUT_BASE_DIR = "."           # 出力先ベースディレクトリ
IMAGE_SIZE = (512, 512)         # リサイズ後の画像サイズ
split_ratios = (0.7, 0.15, 0.15) # Train/Val/Testの分割比率
```

## 📊 出力形式

### U-Net モード
- **画像**: `.npy` (512x512x1, float32, 0-1正規化)
- **ラベル**: `.npy` (512x512x1, バイナリマスク)

### YOLO モード
- **画像**: `.png` (512x512x3, BGR)
- **ラベル**: `.txt` (各行: `class_id x_center y_center width height`, 0-1正規化)
- **設定ファイル**: `data.yaml` (YOLOv5/v8用)

### Faster R-CNN モード
- **画像**: `.npy` (512x512x1, float32, 0-1正規化)
- **ラベル**: `.npy` (Nx5, `[x_min, y_min, x_max, y_max, class_id]`, リサイズ後の座標)

## 🔧 各コンポーネントの説明

### XRayPreprocessor.py
DICOM画像の前処理を担当：
- **DICOM読み込み**: PhotometricInterpretationに基づくモノクロ反転対応
- **正規化**: Min-Max Scaling (0-1範囲)
- **CLAHE**: コントラスト制限付き適応ヒストグラム均等化
- **リサイズ**: 指定サイズへの変換
- **（オプション）アンシャープマスク**: エッジ強調

### LabelProcessor.py
アノテーションデータの処理：
- **U-Net**: マスク画像の読み込みとリサイズ
- **YOLO**: 正規化済み座標の読み込み
- **Faster R-CNN**: 絶対座標の読み込みとスケーリング
- 結節なしデータに対する空ラベル生成

### DataBuilder.py
データセット構築の統合管理：
- ディレクトリ構造の自動生成
- 結節あり/なしデータの収集
- Train/Val/Testへの分割（層化サンプリング）
- バッチ処理とプログレス表示

## 📋 データ分割戦略

1. **結節ありデータ**と**結節なしデータ**を個別にシャッフル
2. それぞれをTrain/Val/Testに分割（デフォルト: 70%/15%/15%）
3. 各split内で結節あり/なしを混合し、再度シャッフル
4. 全splitに両方のクラスが含まれることを保証

## 🎨 画像前処理のカスタマイズ

`XRayPreprocessor.py`では、医療画像に特化した様々な前処理を適用できます。実験目的や使用するモデルに応じて、処理の有効化/無効化やパラメータ調整が可能です。

### 利用可能な前処理メソッド

#### 1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
コントラスト制限付き適応ヒストグラム均等化。局所的なコントラストを強調し、結節の視認性を向上させます。

```python
def apply_clahe(self, img):
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_uint8)
    return img_eq.astype(np.float32) / 255.0
```

**パラメータ調整例:**
```python
# より強いコントラスト強調
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# より細かいタイル分割（局所的な適応）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
```

**効果:**
- 暗い領域と明るい領域の両方で詳細が見やすくなる
- 結節と周辺組織のコントラストが向上
- 過剰な増幅を防ぐclipLimitで自然な見た目を維持

#### 2. **アンシャープマスク (Unsharp Masking)**
エッジを強調し、結節の輪郭をより鮮明にします。

```python
def apply_unsharp_mask(self, img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, np.ones(sharpened.shape))
    return sharpened
```

**パラメータ調整例:**
```python
# より強い鮮鋭化
img = self.apply_unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5)

# より滑らかな鮮鋭化（ノイズ増幅を抑制）
img = self.apply_unsharp_mask(img, kernel_size=(7, 7), sigma=2.0, amount=0.8)
```

**パラメータの意味:**
- `kernel_size`: ぼかしのカーネルサイズ（大きいほど滑らか）
- `sigma`: ガウシアンブラーの標準偏差（大きいほどぼかしが強い）
- `amount`: 鮮鋭化の強度（1.0が標準、大きいほど強調）
- `threshold`: 適用する最小差分（通常は0）

#### 3. **Min-Max正規化**
画像の輝度値を0-1の範囲に正規化します（必須処理）。

```python
def normalize(self, img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min == 0:
        return img
    return (img - img_min) / (img_max - img_min)
```

### 前処理パイプラインのカスタマイズ

`run()`メソッド内で処理順序や有効/無効を調整できます：

```python
def run(self, dicom_path):
    # 1. DICOM読み込み（必須）
    img = self.read_dicom(dicom_path)
    if img is None: return None

    # 2. 正規化（必須）
    img = self.normalize(img)

    # 3. リサイズ（必須）
    img = self.resize(img)

    # 4. CLAHE適用（オプション - コメントアウトで無効化）
    img = self.apply_clahe(img)

    # 5. アンシャープマスク（オプション - デフォルトは無効）
    # img = self.apply_unsharp_mask(img, amount=1.2)

    # 6. チャンネル次元追加（必須）
    img = np.expand_dims(img, axis=-1)

    return img
```

### 推奨される前処理パターン

#### パターン1: 標準設定（汎用性重視）
```python
img = self.normalize(img)
img = self.resize(img)
img = self.apply_clahe(img)  # 有効
# アンシャープマスクは無効
```
**用途:** U-Net、初期実験、バランスの取れた画質

#### パターン2: 最小前処理（モデル依存の学習）
```python
img = self.normalize(img)
img = self.resize(img)
# CLAHEもアンシャープマスクも無効
```
**用途:** データ拡張を使用する場合、モデルに前処理を学習させたい場合

#### パターン3: 最大強調（結節検出重視）
```python
img = self.normalize(img)
img = self.resize(img)
img = self.apply_clahe(img)  # 有効
img = self.apply_unsharp_mask(img, amount=1.2)  # 有効
```
**用途:** 小さな結節の検出、Faster R-CNN、YOLO

#### パターン4: カスタム強調
```python
img = self.normalize(img)
img = self.resize(img)
# より強いCLAHE
img_uint8 = (img * 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(16, 16))
img = clahe.apply(img_uint8).astype(np.float32) / 255.0
# 控えめなアンシャープマスク
img = self.apply_unsharp_mask(img, kernel_size=(7, 7), sigma=1.5, amount=0.8)
```
**用途:** 特定のデータセットに最適化したい場合

### 追加可能な前処理（実装例）

必要に応じて以下の処理も追加できます：

```python
# ガンマ補正
def apply_gamma_correction(self, img, gamma=1.2):
    return np.power(img, gamma)

# ノイズ除去
def apply_denoising(self, img):
    img_uint8 = (img * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(img_uint8, h=10)
    return denoised.astype(np.float32) / 255.0

# モルフォロジー演算
def apply_morphology(self, img, operation='close'):
    img_uint8 = (img * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if operation == 'close':
        result = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
    return result.astype(np.float32) / 255.0
```

### ⚠️ 前処理に関する注意事項

- **処理順序が重要**: 通常は `正規化 → リサイズ → CLAHE → アンシャープマスク` の順
- **過度な処理は逆効果**: 強すぎる前処理はノイズを増幅し、モデルの性能を低下させる可能性
- **実験的評価**: 各前処理の効果は必ずValidationセットで定量評価すること
- **処理時間**: CLAHEとアンシャープマスクは計算コストが高いため、大量データでは処理時間が増加

## 📦 必要なライブラリ

```bash
pip install numpy opencv-python pydicom scikit-learn pyyaml tqdm
```

## ⚠️ 注意事項

- DICOMファイルとラベルファイルはベース名で対応づけられます
- 結節なしデータはサブディレクトリを再帰的に検索します
- ラベルファイルが存在しない結節ありデータは、結節なしとして扱われます
- YOLO形式では空の`.txt`ファイルが結節なし画像用に生成されます
- 出力ディレクトリは実行時に自動的にクリーンアップされます

## 📝 ラベル形式の詳細

### YOLO形式 (*.txt)
```
0 0.512 0.487 0.234 0.198
```
`class_id x_center y_center width height` (全て0-1正規化)

### Faster R-CNN形式 (*.txt)
```
245 312 489 567 0
```
`x_min y_min x_max y_max class_id` (2048x2048原画像の絶対座標)

## 🔍 トラブルシューティング

**Q: "No module named 'PreProcessed'" エラーが出る**  
A: プロジェクトルートから`python PreProcessed/DataBuilder.py`で実行してください

**Q: 一部の画像が処理されない**  
A: DICOMファイルの読み込みに失敗している可能性があります。コンソール出力のエラーメッセージを確認してください

**Q: メモリ不足エラーが出る**  
A: `IMAGE_SIZE`を小さくするか、バッチサイズを調整してください

**Q: 前処理の効果が分からない**  
A: 処理前後の画像を保存して視覚的に比較するか、Validationセットでモデルの性能を評価してください

## 📄 ライセンス

このプロジェクトは医療研究および教育目的で使用されることを想定しています。

## 👤 Author

Kota - University of Tokyo