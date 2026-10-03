"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _softmax(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Tính softmax ổn định số học trên mảng numpy hoặc tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def predict_logits(
    model: nn.Module,
    loader: Iterable,
    device: torch.device | str,
    view: Callable[[torch.Tensor], torch.Tensor] | None = None,
    use_amp: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames, y_true, logits[N, 9]).
    """
    model.eval()
    device = torch.device(device)
    filenames: List[str] = []
    y_true_list: List[int] = []
    logits_list: List[np.ndarray] = []

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = use_amp and device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            # Batch có thể là (images, targets, fnames)
            if len(batch) == 3:
                images, targets, fnames = batch
            elif len(batch) == 2:
                images, targets = batch
                fnames = [f"img_{i}" for i in range(len(targets))]
            else:
                raise ValueError(f"Batch format unexpected: {len(batch)} elements")

            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                out = model(images)

            filenames.extend(list(fnames))
            if isinstance(targets, torch.Tensor):
                y_true_list.extend(targets.cpu().numpy().tolist())
            else:
                y_true_list.extend(list(targets))
            logits_list.append(out.detach().cpu().numpy())

    all_logits = np.concatenate(logits_list, axis=0).astype(np.float32)
    all_y = np.array(y_true_list, dtype=np.int64)
    return filenames, all_y, all_logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên batch ảnh (Identity)."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) bằng torch.flip trên chiều ngang W."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> List[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop` từ batch (N, C, H, W).

    Trả về list gồm 5 batch: [top_left, top_right, bottom_left, bottom_right, center].
    """
    _, _, h, w = x.shape
    if h < crop or w < crop:
        raise ValueError(f"Kích thước ảnh ({h}, {w}) nhỏ hơn crop={crop}")

    top_left = x[..., :crop, :crop]
    top_right = x[..., :crop, w - crop :]
    bottom_left = x[..., h - crop :, :crop]
    bottom_right = x[..., h - crop :, w - crop :]

    h_start = (h - crop) // 2
    w_start = (w - crop) // 2
    center = x[..., h_start : h_start + crop, w_start : w_start + crop]

    return [top_left, top_right, bottom_left, bottom_right, center]


def views_multiscale(x: torch.Tensor, sizes: Sequence[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`.

    Trả về list các batch resized tương ứng.
    """
    out = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        out.append(resized)
    return out


def aggregate_views(
    logits_per_view: Sequence[np.ndarray] | np.ndarray,
    space: str = "prob",
) -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một ma trận xác suất (N, 9) đã chuẩn hoá.

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    if isinstance(logits_per_view, (list, tuple)):
        stacked = np.stack(logits_per_view, axis=0)  # (K, N, 9)
    else:
        stacked = logits_per_view

    if space == "prob":
        probs_per_view = np.stack([_softmax(stacked[k]) for k in range(stacked.shape[0])], axis=0)
        mean_prob = np.mean(probs_per_view, axis=0)
    elif space == "logit":
        mean_logit = np.mean(stacked, axis=0)
        mean_prob = _softmax(mean_logit)
    else:
        raise ValueError(f"Không hỗ trợ space={space}. Chọn 'prob' hoặc 'logit'.")

    # Chuẩn hoá tổng hàng = 1
    mean_prob = mean_prob / np.sum(mean_prob, axis=-1, keepdims=True)
    return mean_prob.astype(np.float64)


def ensemble_probs(list_of_probs: Sequence[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    if len(list_of_probs) == 0:
        raise ValueError("list_of_probs không được rỗng!")
    stacked = np.stack(list_of_probs, axis=0)
    avg_p = np.mean(stacked, axis=0)
    avg_p = avg_p / np.sum(avg_p, axis=-1, keepdims=True)
    return avg_p.astype(np.float64)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69).

    Dùng PyTorch L-BFGS trên tham số log(T).
    Accuracy không đổi vì thứ tự logit không đổi. KHÔNG khớp T trên test.
    """
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.long)

    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=100, line_search_fn="strong_wolfe")

    def _eval_loss():
        optimizer.zero_grad()
        t = torch.exp(log_temp)
        loss = F.cross_entropy(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(_eval_loss)
    best_t = float(torch.exp(log_temp).detach().item())
    # Giới hạn hợp lý để tránh suy biến
    best_t = max(0.01, min(best_t, 10.0))
    return best_t


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T) chuẩn hoá với tổng hàng bằng 1."""
    if T <= 0:
        raise ValueError(f"Nhiệt độ T phải > 0, nhận được T={T}")
    scaled = logits / float(T)
    probs = _softmax(scaled)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return probs.astype(np.float64)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), trả về bản sao không đổi.
    """
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    def _fuse_children(module: nn.Module):
        prev_name = None
        prev_module = None
        for name, child in list(module.named_children()):
            if isinstance(prev_module, nn.Conv2d) and isinstance(child, nn.BatchNorm2d):
                fused = fuse_conv_bn_eval(prev_module, child)
                setattr(module, prev_name, fused)
                setattr(module, name, nn.Identity())
                prev_module = None
                prev_name = None
            else:
                prev_name = name
                prev_module = child
                _fuse_children(child)

    _fuse_children(fused_model)
    return fused_model
