"""Функции потерь под сильный дисбаланс классов (см. постановку кейса:
0.035% пикселей горения на AF, неравномерное 40/37/23% распределение
степеней поражения на BS).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss для бинарной сегментации с сильным дисбалансом (AF)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce

        if valid is not None:
            loss = loss * valid
            return loss.sum() / valid.sum().clamp_min(1.0)
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    """Soft Dice для бинарной маски — напрямую оптимизирует то, что близко к IoU/F1 из метрики."""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        p = torch.sigmoid(logits)
        if valid is not None:
            p = p * valid
            target = target * valid
        dims = tuple(range(1, p.dim()))
        intersection = (p * target).sum(dim=dims)
        union = p.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class AFLoss(nn.Module):
    """Focal + Dice для модуля активного горения."""

    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = BinaryDiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, target, valid) + self.dice_weight * self.dice(logits, target, valid)


class MulticlassDiceLoss(nn.Module):
    """Soft Dice, усреднённый по классам (включая фон) — под 4 класса BS."""

    def __init__(self, n_classes: int, eps: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        if valid is not None:
            v = valid.unsqueeze(1)
            probs = probs * v
            target_oh = target_oh * v

        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice_per_class = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice_per_class.mean()


class BSLoss(nn.Module):
    """Weighted CrossEntropy + multiclass Dice для модуля гарей/степени поражения."""

    def __init__(self, class_weights: list[float], ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.dice = MulticlassDiceLoss(n_classes=len(class_weights))
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weights, reduction="none")
        if valid is not None:
            ce = (ce * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            ce = ce.mean()
        return self.ce_weight * ce + self.dice_weight * self.dice(logits, target, valid)
