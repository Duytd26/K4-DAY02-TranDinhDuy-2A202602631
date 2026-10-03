"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def bench(
    fn: Callable[[], any],
    warmup: int = 10,
    iters: int = 100,
    sync: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo iters lần
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_arr = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "mean": float(np.mean(times_arr)),
        "n": iters,
    }


def latency_report(
    model: nn.Module,
    batch_size: int,
    img_size: int,
    dtype: str = "fp32",
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 100,
) -> Dict[str, any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}
    """
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    is_cuda = dev.type == "cuda"
    sync_fn = torch.cuda.synchronize if is_cuda else None
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    model = model.to(dev)
    model.eval()

    dtype = dtype.lower()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model = model.half()
        x = x.half()

        def _forward():
            with torch.inference_mode():
                return model(x)

    elif dtype == "amp":
        model = model.float()
        x = x.float()

        def _forward():
            with torch.inference_mode():
                with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=is_cuda):
                    return model(x)

    elif dtype == "fp32":
        model = model.float()
        x = x.float()

        def _forward():
            with torch.inference_mode():
                return model(x)
    else:
        raise ValueError(f"Không hỗ trợ dtype={dtype}. Chọn 'fp32', 'amp', hoặc 'fp16'.")

    res = bench(_forward, warmup=warmup, iters=iters, sync=sync_fn)
    p50 = res["p50"]
    images_per_s = float(batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype.upper(),
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(p50, 3),
        "p95": round(res["p95"], 3),
        "p99": round(res["p99"], 3),
        "mean": round(res["mean"], 3),
        "images_per_s": round(images_per_s, 2),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int, **kw) -> Dict[str, any]:
    """Đo độ trễ của TTA K view: so sánh thực tế với K * p50."""
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    dtype = kw.get("dtype", "fp32")
    device = kw.get("device", "cuda")
    warmup = kw.get("warmup", 10)
    iters = kw.get("iters", 50)

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    is_cuda = dev.type == "cuda"
    sync_fn = torch.cuda.synchronize if is_cuda else None

    # Đo đơn view
    single_res = latency_report(model, batch_size=batch_size, img_size=img_size, dtype=dtype,
                                device=device, warmup=warmup, iters=iters)
    single_p50 = single_res["p50"]

    # Đo k-views
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model = model.half()
        x = x.half()
    else:
        model = model.float()
        x = x.float()

    def _forward_k():
        with torch.inference_mode():
            for _ in range(k_views):
                _ = model(x)

    tta_res = bench(_forward_k, warmup=warmup, iters=iters, sync=sync_fn)
    measured_p50 = tta_res["p50"]
    theoretical_p50 = single_p50 * k_views

    return {
        "k_views": k_views,
        "single_view_p50": single_p50,
        "measured_p50": round(measured_p50, 3),
        "theoretical_p50": round(theoretical_p50, 3),
        "ratio": round(measured_p50 / max(single_p50, 1e-6), 2),
    }
