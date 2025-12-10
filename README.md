# Medical Image Dataset Builder for Lung Nodule Detection

肺結節検出のための医療画像データセット構築ツール。JSRT胸部X線データセットから、U-Net、YOLO、Faster R-CNNの各モデル用にデータセットを自動生成します。

## 📋 プロジェクト概要

このプロジェクトは2つのステップで構成されています：

1. **GenerateLabels**: JSRTデータセットの臨床データから各種深層学習モデル用のアノテーションラベルを生成
2. **PreProcessed**: DICOM画像とラベルから学習用データセット（Train/Val/Test）を構築

## 🚀 クイックスタート

### ステップ1: データのセットアップ

プロジェクトのルートディレクトリで、以下のファイルとフォルダを配置してください：

```
project_root/
├── Data/
│   ├── JSRT_DB_ClinicalData_0613_2018_2.csv  # または .xlsx
│   ├── nodule/
│   │   └── DICOM/                # Nodule154imagesフォルダの名前を変更
│   │       └── *.dcm
│   └── non_nodule/
│       └── DICOM/                # NonNodule93imagesフォルダの名前を変更
│           └── *.dcm
```

**重要な初期設定:**
1. `JSRT_DB_ClinicalData_0613_2018_2.csv`（または`.xlsx`）を`Data/`フォルダに配置
2. `Nodule154images`フォルダを`Data/nodule/DICOM/`にリネームして配置
3. `NonNodule93images`フォルダを`Data/non_nodule/DICOM/`にリネームして配置

### ステップ2: ラベル生成（GenerateLabels）

最初に、臨床データからアノテーションラベルを生成します：

```bash
# Faster R-CNN用ラベル生成
python GenerateLabels/gen_F-RCNN_label.py

# YOLO用ラベル生成
python GenerateLabels/gen_YOLO_label.py

# U-Net用マスク生成
python GenerateLabels/gen_U-Net_label.py
```

ラベルは`Data/Labels/`以下に生成されます：
```
Data/
└── Labels/
    ├── F-RCNN/     # Faster R-CNN用テキストラベル
    ├── YOLO/       # YOLO用テキストラベル
    └── U-Net/      # U-Net用マスク画像
```

### ステップ3: 学習データセット構築（PreProcessed）

ラベル生成後、学習用データセットを構築します：

```bash
python PreProcessed/DataBuilder.py
```

出力データセットは以下のように生成されます：
```
[OUTPUT]/
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

## 📁 完全なプロジェクト構造

```
project_root/
├── Data/
│   ├── JSRT_DB_ClinicalData_0613_2018_2.csv
│   ├── nodule/
│   │   ├── DICOM/              # 結節ありDICOM画像（Nodule154images）
│   │   │   └── *.dcm
│   │   └── Labels/
│   │       ├── U-Net/          # セグメンテーションマスク
│   │       │   └── *_mask.png
│   │       ├── YOLO/           # YOLO形式アノテーション
│   │       │   └── *.txt
│   │       └── F-RCNN/         # Faster R-CNN形式アノテーション
│   │           └── *.txt
│   └── non_nodule/
│       └── DICOM/              # 結節なしDICOM画像（NonNodule93images）
│           └── *.dcm
│
├── GenerateLabels/
│   ├── gen_F-RCNN_label.py     # Faster R-CNNラベル生成
│   ├── gen_YOLO_label.py       # YOLOラベル生成
│   └── gen_U-Net_label.py      # U-Netマスク生成
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
```

## 🔧 設定とカスタマイズ

### モード設定（PreProcessed/DataBuilder.py）

`LABEL_MODE`を変更して出力形式を選択：

```python
LABEL_MODE = 'unet'        # U-Net用（セグメンテーションマスク）
LABEL_MODE = 'yolo'        # YOLO用（正規化バウンディングボックス）
LABEL_MODE = 'faster_rcnn' # Faster R-CNN用（絶対座標）
```

### カスタマイズ可能なパラメータ

**DataBuilder.py:**
```python
INPUT_ROOT = "./Data"           # 入力データのルート
OUTPUT_BASE_DIR = "."           # 出力先ベース
IMAGE_SIZE = (512, 512)         # リサイズ後の画像サイズ
split_ratios = (0.7, 0.15, 0.15) # Train/Val/Testの分割比率
```

**ラベル生成スクリプト:**
```python
PIXEL_SPACING = 0.175  # mm/pixel（JSRTデータセット標準値）
```

## 📊 出力形式の詳細

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

## 🎨 画像前処理のカスタマイズ

`XRayPreprocessor.py`では、医療画像に特化した前処理を適用できます：

### 利用可能な前処理メソッド

#### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
局所的なコントラストを強調し、結節の視認性を向上：

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

# より細かいタイル分割
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
```

#### 2. アンシャープマスク
エッジを強調し、結節の輪郭をより鮮明に：

```python
def apply_unsharp_mask(self, img, kernel_size=(5, 5), sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, np.ones(sharpened.shape))
    return sharpened
```

### 推奨される前処理パターン

**パターン1: 標準設定（汎用性重視）**
```python
img = self.normalize(img)
img = self.resize(img)
img = self.apply_clahe(img)  # 有効
# アンシャープマスクは無効
```

**パターン2: 最小前処理**
```python
img = self.normalize(img)
img = self.resize(img)
# CLAHEもアンシャープマスクも無効
```

**パターン3: 最大強調（結節検出重視）**
```python
img = self.normalize(img)
img = self.resize(img)
img = self.apply_clahe(img)
img = self.apply_unsharp_mask(img, amount=1.2)
```

## 📋 データ分割戦略

1. **結節ありデータ**と**結節なしデータ**を個別にシャッフル
2. それぞれをTrain/Val/Testに分割（デフォルト: 70%/15%/15%）
3. 各split内で結節あり/なしを混合し、再度シャッフル
4. 全splitに両方のクラスが含まれることを保証

## 📦 必要なライブラリ

```bash
pip install numpy opencv-python pydicom scikit-learn pyyaml tqdm pandas
```

## 🔍 トラブルシューティング

### データセットアップに関するエラー

**Q: "CSVファイルが見つかりません"**  
A: `Data/JSRT_DB_ClinicalData_0613_2018_2.csv`（または`.xlsx`）が正しく配置されているか確認

**Q: "DICOMファイルが見つかりません"**  
A: 以下を確認：
- `Data/nodule/DICOM/`に`Nodule154images`の内容が配置されているか
- `Data/non_nodule/DICOM/`に`NonNodule93images`の内容が配置されているか

### ラベル生成に関するエラー

**Q: "No module named 'PreProcessed'" エラー**  
A: プロジェクトルートから実行してください：`python PreProcessed/DataBuilder.py`

**Q: "一部の画像が処理されない"**  
A: DICOMファイルの読み込みに失敗している可能性があります。コンソール出力のエラーメッセージを確認

### メモリとパフォーマンス

**Q: "メモリ不足エラー"**  
A: `IMAGE_SIZE`を小さくするか、バッチサイズを調整してください

**Q: "処理が遅い"**  
A: CLAHEとアンシャープマスクは計算コストが高いため、必要に応じて無効化を検討

## ⚠️ 重要な注意事項

- DICOMファイルとラベルファイルはベース名で対応づけられます
- ラベル生成を**必ず先に**実行してから、データセット構築を行ってください
- 出力ディレクトリは実行時に自動的にクリーンアップされます
- 前処理の効果はValidationセットで定量評価することを推奨します

## 📖 技術仕様

### 入力データ
- **画像サイズ**: 2048×2048 pixels
- **座標系**: 左上原点、右下が(2047, 2047)
- **ピクセル間隔**: 0.175 mm/pixel（JSRTデータセット標準）

### クラスマッピング
- **Faster R-CNN**: benign=1, malignant=2
- **YOLO**: benign=0, malignant=1
- **U-Net**: 結節領域=255（白）、背景=0（黒）

## 参考文献

- JSRT Database: http://db.jsrt.or.jp/
- Faster R-CNN: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection"
- YOLO: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"

## 👤 Author

Kota - University of Tokyo
