"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Đảm bảo import được các module trong repo
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import dataset  # noqa: E402
import eval as ev  # noqa: E402
import losses  # noqa: E402
import model as model_module  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (model.param_groups)."""
    groups = model_module.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name])
        self.original.clear()


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    if cfg.init == "frozen":
        model_module.freeze_backbone(model)
        # Giữ BatchNorm của backbone ở eval mode khi đóng băng
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and m != model.get_classifier():
                m.eval()

    total_loss = 0.0
    num_samples = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.size(0)

        if cfg.mix in ("mixup", "cutmix"):
            x_mixed, targets_mixed = losses.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
            optimizer.zero_grad()
            if cfg.amp and device.type == "cuda":
                with autocast():
                    logits = model(x_mixed)
                    loss = losses.mixed_loss(criterion, logits, targets_mixed)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_mixed)
                loss = losses.mixed_loss(criterion, logits, targets_mixed)
                loss.backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            if cfg.amp and device.type == "cuda":
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / max(1, num_samples), "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for x, y, fnames in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_size = x.size(0)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * batch_size
            num_samples += batch_size

            all_filenames.extend(fnames)
            all_targets.append(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, ev.NUM_CLASSES))
    avg_loss = total_loss / max(1, num_samples)
    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    val_acc = [h["val_top1"] for h in history]
    lrs = [h.get("lr", 0.0) for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Loss
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_loss, "r-s", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Metrics
    ax2.plot(epochs, val_f1, "g-^", label="Val Macro-F1", markersize=4)
    ax2.plot(epochs, val_acc, "m-d", label="Val Top-1 Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{title} - Val Metrics")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    # LR schedule
    ax3.plot(epochs, lrs, "c-", label="LR")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("LR Schedule")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    set_seed(cfg.seed)
    save_dir = run_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)

    # Ghi config.json
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 1. Dataset & Split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False, num_workers=cfg.num_workers
    )

    # 2. Model & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_module.build_model(
        cfg.backbone, pretrained=True, num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init
    ).to(device)

    # 3. Loss
    crit_kwargs = {}
    if cfg.loss == "ls":
        crit_kwargs["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        crit_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().values
        crit_kwargs["weight"] = losses.class_weights(counts, beta=cfg.class_weight_beta or 0.0).to(device)

    criterion = losses.build_criterion(cfg.loss, **crit_kwargs)

    # 4. Optimizer, Scheduler, AMP, EMA
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Loop
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_model_weights = None
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        t_epoch = time.perf_counter() - t0
        epoch_times.append(t_epoch)

        # Đánh giá bằng EMA nếu có
        if ema is not None:
            ema.apply_shadow(model)
        val_fnames, val_y, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        if ema is not None:
            ema.restore(model)

        val_preds = val_logits.argmax(axis=-1)
        val_metrics = ev.compute_metrics(val_y, val_preds)

        row = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top1": val_metrics["top1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "lr": train_res["lr"],
            "time_s": t_epoch
        }
        history.append(row)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            if ema is not None:
                ema.apply_shadow(model)
            best_model_weights = copy.deepcopy(model.state_dict())
            if ema is not None:
                ema.restore(model)
            torch.save(best_model_weights, save_dir / "best_model.pt")

    # Lưu history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(save_dir / "history.csv", index=False)

    curve_path = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} ({cfg.backbone})")

    # 6. Load best model và lưu val predictions
    model.load_state_dict(best_model_weights)
    val_fnames, val_y, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = softmax(val_logits)
    ev.save_predictions(pred_path(cfg, "val"), val_fnames, val_y, val_probs)
    np.save(save_dir / "val_logits.npy", val_logits)

    test_macro_f1 = None
    test_top1 = None
    test_ece = None

    # 7. Nếu bật save_test_predictions (Bước 4): đánh giá test
    if cfg.save_test_predictions:
        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tf, cfg.batch_size, train=False, num_workers=cfg.num_workers
        )
        test_fnames, test_y, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = softmax(test_logits)
        ev.save_predictions(pred_path(cfg, "test"), test_fnames, test_y, test_probs)
        np.save(save_dir / "test_logits.npy", test_logits)

        test_preds = test_probs.argmax(axis=-1)
        tm = ev.compute_metrics(test_y, test_preds)
        test_macro_f1 = tm["macro_f1"]
        test_top1 = tm["top1"]
        test_ece = ev.compute_ece(test_y, test_probs)

    num_params = model_module.count_params(model)
    gmacs = model_module.count_gmacs(model, cfg.img_size)

    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "val_top1": hist_df.loc[best_epoch - 1, "val_top1"],
        "test_macro_f1": test_macro_f1,
        "test_top1": test_top1,
        "test_ece": test_ece,
        "avg_epoch_time_s": float(np.mean(epoch_times)),
        "params_m": num_params,
        "gmacs": gmacs,
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    defaults = Config()
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Tham số không hợp lệ (thiếu =): {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not hasattr(defaults, k):
            raise KeyError(f"Trường '{k}' không tồn tại trong Config!")

        default_val = getattr(defaults, k)
        if default_val is None:
            if v.lower() in ("none", "null", ""):
                out[k] = None
            elif v.replace(".", "", 1).isdigit():
                out[k] = float(v) if "." in v else int(v)
            else:
                out[k] = v
        elif isinstance(default_val, bool):
            out[k] = v.lower() in ("true", "1", "yes")
        elif isinstance(default_val, int):
            out[k] = int(v)
        elif isinstance(default_val, float):
            out[k] = float(v)
        else:
            out[k] = None if v.lower() in ("none", "null") else v
    return out


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="DeepWeeds Training")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè cấu hình: key=val ...")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"=== Chạy thí nghiệm {cfg.exp_id} (Backbone: {cfg.backbone}, Seed: {cfg.seed}) ===")
    res = run(cfg)
    print("=== Hoàn thành ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
