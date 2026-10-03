"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None: đúng bằng Cross-Entropy thông thường.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # None hoặc tensor trọng số
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
    """
    counts = np.array(counts, dtype=np.float64)
    if beta <= 0.0:
        w = 1.0 / counts
        w = w / w.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / effective_num
        w = w / w.sum() * len(counts)
    return torch.as_tensor(w, dtype=torch.float)


def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix)."""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[perm, :, bbx1:bbx2, bby1:bby2]
        box_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = x.size(2) * x.size(3)
        actual_lam = 1.0 - (box_area / float(total_area))
        return x_mixed, (y_a, y_b, actual_lam)
    else:
        return x, (y_a, y_b, 1.0)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
