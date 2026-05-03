"""Sample 250 chips from a Sen1Floods11 release into images/ + labels/ folders."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import random
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_images", required=True, help="Dir of *.tif or *.png S2 RGB chips")
    ap.add_argument("--src_labels", required=True, help="Dir of *.tif or *.png label masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src_imgs = sorted(Path(args.src_images).iterdir())
    paired = []
    for ip in src_imgs:
        if ip.suffix.lower() not in (".png", ".jpg", ".tif", ".tiff"):
            continue
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            mp = Path(args.src_labels) / (ip.stem + ext)
            if mp.exists():
                paired.append((ip, mp))
                break
    print(f"Found {len(paired)} paired chips.")
    sampled = random.sample(paired, min(args.n, len(paired)))

    out_imgs = Path(args.out) / "images"
    out_lbls = Path(args.out) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    for ip, mp in sampled:
        # Convert to PNG, resize/normalize-on-load handled by FSSD transforms.
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(out_imgs / (ip.stem + ".png")), img)
        msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        msk = (msk > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_lbls / (mp.stem + ".png")), msk)
    print(f"Wrote {len(sampled)} pairs to {args.out}")


if __name__ == "__main__":
    main()
