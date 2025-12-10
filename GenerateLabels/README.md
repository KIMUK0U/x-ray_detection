# JSRT Medical Imaging Label Generator

JSRTデータセット用の物体検出・セグメンテーションラベル生成ツール

## 概要

このプロジェクトは、JSRT（Japanese Society of Radiological Technology）胸部X線データセットの臨床データから、各種深層学習モデル用のアノテーションラベルを自動生成するツール群です。

以下の3つの主要な深層学習フレームワークに対応したラベル形式を生成できます：

- **Faster R-CNN**: 物体検出用のバウンディングボックス座標
- **YOLO**: 正規化された物体検出ラベル
- **U-Net**: セグメンテーション用のマスク画像

## 特徴

- 医療画像特有のピクセル間隔（pixel spacing）を考慮したミリメートル→ピクセル変換
- 良性・悪性結節の分類ラベル対応
- 2048×2048サイズの胸部X線画像に最適化
- クラスベースの実装で拡張性が高い
- エラーハンドリングとデータ検証機能

## 必要な環境

```
Python 3.7以上
```

### 依存ライブラリ

```bash
pip install pandas numpy opencv-python
```

または、以下のコマンドで一括インストール：

```bash
pip install -r requirements.txt
```

## ディレクトリ構成

```
project/
├── gen_F-RCNN_label.py      # Faster R-CNN用ラベル生成
├── gen_YOLO_label.py         # YOLO用ラベル生成
├── gen_U-Net_label.py        # U-Net用マスク生成
├── Data/
│   ├── JSRT_DB_ClinicalData_0613_2018_2.csv  # 入力CSVファイル
│   └── Labels/
│       ├── F-RCNN/           # Faster R-CNN出力先
│       ├── YOLO/             # YOLO出力先
│       └── U-Net/            # U-Net出力先
└── README.md
```

## 入力データ形式

CSVファイルには以下のカラムが必要です：

| カラム名 | 説明 | 単位 |
|---------|------|------|
| `ImageFileName` | 画像ファイル名 | - |
| `benign/ malignant` | 結節の分類 | benign/malignant |
| `Size[mm]` | 結節の直径 | mm |
| `X-cor` | 結節中心のX座標 | pixel |
| `Y-cor` | 結節中心のY座標 | pixel |

## 使用方法

### 1. Faster R-CNN用ラベル生成

```bash
cd x-ray_detection
python GenerateLabels/gen_F-RCNN_label.py
```

**出力形式** (テキストファイル: `{画像名}.txt`)：
```
x_min y_min x_max y_max class_id
```

- `class_id`: 1=良性（benign）, 2=悪性（malignant）
- 座標は整数値（0〜2047の範囲）

**例**：
```
512 768 612 868 1
1024 1536 1124 1636 2
```

### 2. YOLO用ラベル生成

```bash
cd x-ray_detection
python GenerateLabels/gen_YOLO_label.py
```

**出力形式** (テキストファイル: `{画像名}.txt`)：
```
class_id x_center_norm y_center_norm w_norm h_norm
```

- `class_id`: 0=良性（benign）, 1=悪性（malignant）
- すべての座標値は0.0〜1.0に正規化
- 座標は画像サイズで正規化された浮動小数点数（小数点以下4桁）

**例**：
```
0 0.2500 0.3750 0.0488 0.0488
1 0.5000 0.7500 0.0488 0.0488
```

### 3. U-Net用マスク生成

```bash
cd x-ray_detection
python GenerateLabels/gen_U-Net_label.py
```

**出力形式** (PNGファイル: `{画像名}_mask.png`)：
- グレースケール画像（2048×2048）
- 結節領域: 255（白）
- 背景: 0（黒）
- 円形の塗りつぶしマスク

## クラス構成

### FasterRCNNLabelGenerator

Faster R-CNN用のバウンディングボックスラベルを生成します。

```python
generator = FasterRCNNLabelGenerator(csv_path, pixel_spacing=0.175)
labels = generator.generate_labels()
```

**主要メソッド**：
- `generate_labels()`: ラベル辞書を生成（画像名→ラベルリスト）
- `_mm_to_pixel()`: mmからピクセルへの変換

### YOLOLabelGenerator

YOLO形式の正規化されたラベルを生成します。

```python
generator = YOLOLabelGenerator(csv_path, pixel_spacing=0.175)
labels = generator.generate_labels()
```

**主要メソッド**：
- `generate_labels()`: 正規化されたYOLOラベルを生成
- `_mm_to_pixel()`: mmからピクセルへの変換

### UNetLabelGenerator

U-Net用のセグメンテーションマスクを生成します。

```python
generator = UNetLabelGenerator(csv_path, pixel_spacing=0.175)
masks = generator.generate_masks()
```

**主要メソッド**：
- `generate_masks()`: マスク画像の辞書を生成
- `_mm_to_radius_pixel()`: mmから半径ピクセルへの変換

## パラメータ設定

各スクリプト内で以下のパラメータを調整できます：

```python
FILE_NAME_CSV = "./Data/JSRT_DB_ClinicalData_0613_2018_2.csv"
PIXEL_SPACING = 0.175  # mm/pixel
OUTPUT_DIR = "./Data/Labels/{モデル名}"
```

### PIXEL_SPACING について

- デフォルト値: `0.175 mm/pixel`
- JSRTデータセットの標準的なピクセル間隔
- 使用する画像の実際のピクセル間隔に合わせて調整してください

## 出力例

### Faster R-CNN ラベル

`JPCLN001.txt`:
```
856 1245 920 1309 1
1456 789 1520 853 2
```

### YOLO ラベル

`JPCLN001.txt`:
```
0 0.4180 0.6084 0.0313 0.0313
1 0.7109 0.3857 0.0313 0.0313
```

### U-Net マスク

`JPCLN001_mask.png`: 2048×2048のグレースケール画像

## トラブルシューティング

### CSVファイルが見つからない

```
FileNotFoundError: エラー: CSVファイルが指定されたパスに見つかりません
```

**解決方法**：
- CSVファイルのパスが正しいか確認
- `FILE_NAME_CSV`変数のパスを実際のファイル位置に修正

### データの欠損

スクリプトは以下の処理で欠損データを自動的に除外します：
- 結節分類（`benign/ malignant`）が空の行
- サイズ・座標情報が数値でない行
- 必須カラムが欠けている行

### 座標の範囲外エラー

- 座標は自動的に`0〜2047`の範囲にクリッピングされます
- YOLOラベルは`0.0〜1.0`に正規化・クリッピングされます

## 技術仕様

- **画像サイズ**: 2048×2048 pixels
- **座標系**: 左上原点、右下が(2047, 2047)
- **結節形状**: 円形（直径から半径を計算）
- **クラスマッピング**:
  - Faster R-CNN: benign=1, malignant=2
  - YOLO: benign=0, malignant=1

## ライセンス

このプロジェクトは教育・研究目的で使用してください。

## 参考文献

- JSRT Database: [http://db.jsrt.or.jp/](http://db.jsrt.or.jp/)
- Faster R-CNN: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection"
- YOLO: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"

## 貢献

バグ報告や機能リクエストは、Issueを作成してください。

## 変更履歴

- **v1.0.0** (2024): 初版リリース
  - Faster R-CNN、YOLO、U-Netラベル生成機能
  - 基本的なエラーハンドリング実装
