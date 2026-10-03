"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    pretrained_flag = pretrained if init != "scratch" else False
    model = timm.create_model(
        name,
        pretrained=pretrained_flag,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head."""
    for param in model.parameters():
        param.requires_grad = False
    head = model.get_classifier()
    for param in head.parameters():
        param.requires_grad = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    head = model.get_classifier()
    head_params = set(head.parameters())

    group_decay = []
    group_no_decay = []
    group_head = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in head_params:
            group_head.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            group_no_decay.append(param)
        else:
            group_decay.append(param)

    return [
        {"params": group_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": group_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": group_head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x)."""
    total_macs = 0
    hooks = []

    def conv_hook(m, inp, out):
        nonlocal total_macs
        if out.ndim >= 4:
            total_macs += out.size(2) * out.size(3) * (m.in_channels // m.groups) * m.out_channels * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        nonlocal total_macs
        total_macs += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_input = torch.zeros(1, 3, img_size, img_size, device=device)
    training_state = model.training
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    if training_state:
        model.train()

    return total_macs / 1e9
