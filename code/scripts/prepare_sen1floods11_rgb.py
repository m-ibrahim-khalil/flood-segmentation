"""Sample 250 chips from a Sen1Floods11 release into images/ + labels/ folders."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import random
import cv2
import numpy as np


def _candidate_label_stems(image_stem: str) -> list[str]:
    """Return stems to try for the matching label, in order.

    Sen1Floods11 names images <event>_<id>_S2Hand.tif but labels
    <event>_<id>_LabelHand.tif. We try the original stem first (works for
    same-stem datasets like FSSD), then the S2Hand→LabelHand substitution,
    then S1Hand→LabelHand for the SAR variant.
    """
    out = [image_stem]
    if "_S2Hand" in image_stem:
        out.append(image_stem.replace("_S2Hand", "_LabelHand"))
    if "_S1Hand" in image_stem:
        out.append(image_stem.replace("_S1Hand", "_LabelHand"))
    return out


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
    paired: list[tuple[Path, Path]] = []
    for ip in src_imgs:
        if ip.suffix.lower() not in (".png", ".jpg", ".tif", ".tiff"):
            continue
        found = None
        for stem in _candidate_label_stems(ip.stem):
            for ext in (".png", ".tif", ".tiff", ".jpg"):
                mp = Path(args.src_labels) / (stem + ext)
                if mp.exists():
                    found = mp
                    break
            if found:
                break
        if found:
            paired.append((ip, found))
    print(f"Found {len(paired)} paired chips.")
    sampled = random.sample(paired, min(args.n, len(paired)))

    out_imgs = Path(args.out) / "images"
    out_lbls = Path(args.out) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    for ip, mp in sampled:
        # Sen1Floods11 S2Hand chips are 13-band Sentinel-2 GeoTIFFs at 10m;
        # we need RGB. Use rasterio when available (handles multi-band TIFFs),
        # fall back to cv2 for single-band images and FSSD-style PNG/JPG.
        try:
            import rasterio
        except Exception:
            rasterio = None

        if rasterio is not None and ip.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(str(ip)) as src:
                if src.count >= 4:
                    # Sentinel-2 band order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
                    # Many Sen1Floods11 dumps use B2(blue),B3(green),B4(red) as bands 2,3,4
                    # but the official S2Hand release follows ESA order (B1=index 1).
                    blue  = src.read(2).astype(np.float32)
                    green = src.read(3).astype(np.float32)
                    red   = src.read(4).astype(np.float32)
                    rgb = np.stack([red, green, blue], axis=-1)
                else:
                    rgb = src.read(1).astype(np.float32)
                    rgb = np.stack([rgb, rgb, rgb], axis=-1)
            # 2-98 percentile stretch — the standard Sentinel-2 visualisation
            lo, hi = np.percentile(rgb, [2, 98])
            rgb = np.clip((rgb - lo) / max(hi - lo, 1e-8), 0, 1)
            img = (rgb * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                n_skipped += 1
                continue
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            if img_bgr.dtype != np.uint8:
                img_bgr = ((img_bgr - img_bgr.min()) /
                           (img_bgr.max() - img_bgr.min() + 1e-8) * 255).astype(np.uint8)

        cv2.imwrite(str(out_imgs / (ip.stem + ".png")), img_bgr)

        # Mask: simple binarisation
        if rasterio is not None and mp.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(str(mp)) as src:
                msk_raw = src.read(1)
        else:
            msk_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if msk_raw is None:
                n_skipped += 1
                continue
        # Sen1Floods11 LabelHand uses {-1: nodata, 0: not water, 1: water}.
        # Treat 1 as flood, everything else as background.
        msk = (msk_raw == 1).astype(np.uint8) * 255
        cv2.imwrite(str(out_lbls / (ip.stem + ".png")), msk)
    if n_skipped:
        print(f"Skipped {n_skipped} unreadable pairs.")
    print(f"Wrote {len(sampled) - n_skipped} pairs to {args.out}")


if __name__ == "__main__":
    main()
