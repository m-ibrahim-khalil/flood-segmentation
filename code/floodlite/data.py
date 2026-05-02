"""FSSD dataset loader with Albumentations transforms.

Expected on-disk layout (matches the Kaggle dataset
`lihuayang111265/flood-semantic-segmentation-dataset`):

    <DATA_ROOT>/
        dataset/
            train/
                images/   *.jpg|*.png
                labels/   *.png|*.jpg
            val/
                images/
                labels/

By default we combine train + val into one set of (image, mask) pairs and
expose a 5-fold cross-validation split (matches Karcı et al. 2026).
Pass ``use_predefined_split=True`` to use the dataset's own train/val
split instead.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS = (".png", ".jpg", ".jpeg")  # tried in this order to find the mask for a given image stem


def get_transforms(img_size: int = 256, train: bool = True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def find_split_dirs(root: str | Path, split: str) -> tuple[Path | None, Path | None]:
    """Locate (images_dir, labels_dir) for split in {'train','val'}.

    Robust to variations in layout casing (images / Image / IMAGES, labels / Mask /
    masks / Labels / mask) so it also works on the older root-level FSSD packaging.
    """
    root = Path(root)
    candidates = (
        list(root.rglob(f"{split}/images"))
        + list(root.rglob(f"{split}/Image"))
        + list(root.rglob(f"{split}/IMAGES"))
    )
    if not candidates:
        return None, None
    img_dir = candidates[0]
    for lbl_name in ("labels", "Mask", "masks", "Labels", "mask"):
        lbl_dir = img_dir.parent / lbl_name
        if lbl_dir.exists():
            return img_dir, lbl_dir
    return img_dir, None


def gather_pairs(img_dir: Path | None, lbl_dir: Path | None) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs whose stems match."""
    pairs: list[tuple[Path, Path]] = []
    if img_dir is None or lbl_dir is None:
        return pairs
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        for ext in MASK_EXTS:
            mp = lbl_dir / (ip.stem + ext)
            if mp.exists():
                pairs.append((ip, mp))
                break
    return pairs


def collect_pairs(root: str | Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Walk a Kaggle-FSSD root and return (train_pairs, val_pairs)."""
    train_imgs, train_lbls = find_split_dirs(root, "train")
    val_imgs, val_lbls = find_split_dirs(root, "val")
    return gather_pairs(train_imgs, train_lbls), gather_pairs(val_imgs, val_lbls)


class FSSD(Dataset):
    """FSSD dataset built from a list of (image_path, mask_path) pairs."""

    def __init__(self, pairs: list[tuple[Path, Path]], indices: Iterable[int] | None = None,
                 img_size: int = 256, train: bool = True):
        self.pairs = [pairs[i] for i in indices] if indices is not None else list(pairs)
        if not self.pairs:
            raise ValueError("FSSD got an empty list of (image, mask) pairs.")
        self.tfm = get_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        out = self.tfm(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)  # (1, H, W)


def make_loaders(root: str | Path, fold: int, n_folds: int = 5,
                 batch_size: int = 8, img_size: int = 256, num_workers: int = 2,
                 seed: int = 42, use_predefined_split: bool = False):
    """Construct train/val DataLoaders.

    If ``use_predefined_split`` is True, the dataset's own train/ and val/
    folders are used (``fold`` is ignored). Otherwise we combine the two
    folders and run ``n_folds``-way cross-validation, returning the loader
    pair for the requested fold.
    """
    train_pairs, val_pairs = collect_pairs(root)
    if not train_pairs and not val_pairs:
        raise FileNotFoundError(
            f"Could not find FSSD train/val image folders under {root!s}. "
            "Expected dataset/{train,val}/{images,labels}."
        )

    if use_predefined_split:
        train_ds = FSSD(train_pairs, img_size=img_size, train=True)
        val_ds = FSSD(val_pairs, img_size=img_size, train=False)
    else:
        all_pairs = train_pairs + val_pairs
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(len(all_pairs))))
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0,{n_folds}); got {fold}")
        train_idx, val_idx = splits[fold]
        train_ds = FSSD(all_pairs, indices=train_idx.tolist(), img_size=img_size, train=True)
        val_ds = FSSD(all_pairs, indices=val_idx.tolist(), img_size=img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
