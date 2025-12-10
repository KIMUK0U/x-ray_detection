import torch.nn.functional as F
import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth # 分母がゼロになるのを防ぐための平滑化項

    def forward(self, inputs, targets):
        # ⚠️ inputsがロジット（BCEWithLogitsLossの出力）であると仮定し、
        # シグモイド関数を適用して確率に変換します。
        # Multi-classの場合はsoftmaxを使用します。
        inputs = torch.sigmoid(inputs)  
        
        # テンソルを平坦化（flatten）します。
        # バッチとチャンネル次元は保持し、それ以外の次元を統合して計算することが一般的です。
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Intersection（交差）
        intersection = (inputs * targets).sum()                            
        
        # Dice Scoreの計算
        # 分母は予測の二乗和とターゲットの二乗和の和を使用すると、
        # 勾配がより安定すると言われています。
        dice_score = (2. * intersection + self.smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + self.smooth)
        
        # Dice Loss = 1 - Dice Score
        return 1. - dice_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # クラス重み付けパラメータ
        self.gamma = gamma  # フォーカスパラメータ
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. BCEWithLogitsLossと同様に、ロジットを入力として受け取ります
        #    そして、シグモイド関数を適用して確率(p)に変換します。
        p = torch.sigmoid(inputs)

        # 2. 標準的なBCE Lossを計算します
        #    BCEWithLogitsLoss(inputs, targets, reduction='none') とほぼ同じ
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # 3. p_t の計算 (正解クラスの確率)
        #    t=1のとき p_t = p, t=0のとき p_t = 1-p となるようにします
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # 4. 変調因子の計算: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        
        # 5. Focal Lossの計算
        #    alphaを適用します: 正例はalpha、負例は(1-alpha)の重み
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_factor * modulating_factor * ce_loss

        # 6. リダクション（平均または合計）
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # reduction='none'
            return focal_loss
        
# 複合損失クラスを定義します
class WeightedCombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.7, focal_weight=0.3):
        super(WeightedCombinedLoss, self).__init__()
        
        # 重みの設定
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # 損失関数のインスタンス化
        self.focal_criterion = FocalLoss(alpha=0.5, gamma=2.0)
        self.dice_criterion = DiceLoss()
        
    def forward(self, inputs, targets):
        
        # 1. BCE With Logits Lossの計算
        # inputsはロジット（sigmoid前の値）を使用します
        focal_loss = self.focal_criterion(inputs, targets)
        
        # 2. Dice Lossの計算
        # inputsはロジット（DiceLoss内でsigmoid処理されます）を使用します
        dice_loss = self.dice_criterion(inputs, targets)
        
        # 3. 重み付けして結合
        combined_loss = (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)
        
        return combined_loss