import torch
import torch.nn as nn
import torch.nn.functional as F

#==========================================
# モデル定義 (U-Net)
# 
# 設計のポイント:
# - 入力画像サイズに応じてダウンサンプリング段数を調整
#   例: 512x512 → 5段階, 1024x1024 → 6段階
# - 各段階でMaxPool2dにより解像度が1/2になる
# - 最小解像度がconv演算可能なサイズ(最低4x4程度)になるよう段数を決定
# - base_channelsは計算量とメモリのバランスを考慮して設定(通常32-64)
# - デコーダのDoubleConv入力チャンネル数はスキップ接続により2倍になる
#==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # WRITE ME: residual接続の条件とconv_blockを定義

    def forward(self, x):
        # WRITE ME: forward処理を実装
        pass


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # WRITE ME: エンコーダとデコーダの各層を定義

    def forward(self, x):
        # WRITE ME: forward処理を実装
        
        logits = self.outc(x)
        return logits