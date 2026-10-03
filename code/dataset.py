"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict số liệu để dán vào báo cáo.
    """
    images_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    # 1. Tỉ lệ từng tập
    pct_train = n_train / n_total * 100
    pct_val = n_val / n_total * 100
    pct_test = n_test / n_total * 100

    # Số lượng theo từng lớp
    per_class_train = train_df["Label"].value_counts().sort_index().to_dict()
    per_class_val = val_df["Label"].value_counts().sort_index().to_dict()
    per_class_test = test_df["Label"].value_counts().sort_index().to_dict()

    # 2. Giao của từng cặp tập theo Filename phải RỖNG
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    overlap_train_val = len(train_files & val_files)
    overlap_train_test = len(train_files & test_files)
    overlap_val_test = len(val_files & test_files)

    assert overlap_train_val == 0, f"LỖI: train và val trùng {overlap_train_val} ảnh!"
    assert overlap_train_test == 0, f"LỖI: train và test trùng {overlap_train_test} ảnh!"
    assert overlap_val_test == 0, f"LỖI: val và test trùng {overlap_val_test} ảnh!"

    # 3. Hợp ba tập phải bằng đúng 17.509 ảnh
    union_files = train_files | val_files | test_files
    assert len(union_files) == 17509, f"LỖI: Hợp 3 tập có {len(union_files)} ảnh, kỳ vọng 17509 ảnh!"

    # 4. Mọi Filename đều tồn tại trong images_dir
    existing_images = set(p.name for p in images_dir.glob("*.jpg"))
    if not existing_images:  # fallback check if glob didn't match
        existing_images = set(p.name for p in images_dir.iterdir() if p.is_file())
    missing_files = union_files - existing_images
    assert len(missing_files) == 0, f"LỖI: Có {len(missing_files)} file ảnh không tồn tại trong {images_dir}!"

    info = {
        "n": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": n_total,
            "percentages": {"train": f"{pct_train:.2f}%", "val": f"{pct_val:.2f}%", "test": f"{pct_test:.2f}%"}
        },
        "per_class": {
            "train": per_class_train,
            "val": per_class_val,
            "test": per_class_test
        },
        "overlap": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test
        },
        "missing_images": len(missing_files)
    }

    print("=== KẾT QUẢ KIỂM TRA CHIA DỮ LIỆU (SPLIT CHECK) ===")
    print(f"1. Số lượng ảnh: Train={n_train} ({pct_train:.2f}%), Val={n_val} ({pct_val:.2f}%), Test={n_test} ({pct_test:.2f}%)")
    print(f"2. Giao giữa các tập: train∩val={overlap_train_val}, train∩test={overlap_train_test}, val∩test={overlap_val_test} (ĐỀU RỖNG)")
    print(f"3. Hợp 3 tập: {len(union_files)} ảnh (BẰNG ĐÚNG 17.509)")
    print(f"4. Kiểm tra file ảnh tồn tại trên đĩa: Không thiếu ảnh nào (Missing={len(missing_files)})")
    print("=> TẤT CẢ KIỂM TRA HỢP LỆ VÀ ĐẠT CHUẨN S1-S6!")
    return info


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation.
    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> Resize(256) + CenterCrop(img_size) + ToTensor + Normalize.
    KHÔNG augmentation ngẫu nhiên khi đánh giá.
    """
    if train:
        t_list = []
        if aug == "basic":
            t_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        elif aug == "color":
            t_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        elif aug == "trivial":
            t_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.TrivialAugmentWide(),
            ])
        elif aug == "randaug":
            t_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ])
        else:
            t_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transforms.Compose(t_list)
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        filename = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / filename
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[l] for l in df["Label"]]
            weighted_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(df),
                replacement=True
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=weighted_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    return loader
