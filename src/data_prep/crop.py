"""Central crop generator for converting UWF images to narrow FOV crops."""
# DONT USE THIS FILE pls
# Pipeline goes from preprocess -> synthetic_crop -> feature_extract
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from PIL import Image


@dataclass
class CropConfig:
    src_root: Path
    dst_root: Path
    crop_deg: float = 50.0
    full_deg: float = 200.0
    labels_name: str = "labels.csv"

    @property
    def ratio(self) -> float:
        r = self.crop_deg / self.full_deg
        return max(0.0, min(1.0, r))


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        print(f"labels.csv not found at {labels_path}; nothing to crop yet.")
        return pd.DataFrame()
    df = pd.read_csv(labels_path)
    if "image_path" not in df.columns or "class" not in df.columns:
        raise ValueError("labels.csv must have columns: image_path,class")
    return df[["image_path", "class"]]


def center_square_crop(img: Image.Image, ratio: float) -> Image.Image:
    ratio = max(0.0, min(1.0, ratio))
    w, h = img.size
    side = int(min(w, h) * ratio)
    if side <= 0:
        raise ValueError("Computed crop size is zero; check ratio/full_deg/crop_deg.")
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def process(config: CropConfig) -> Tuple[int, List[str]]:
    df = load_labels(config.src_root / config.labels_name)
    if df.empty:
        return 0, []

    out_img_dir = config.dst_root / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    skipped: List[str] = []
    rows = []

    for _, row in df.iterrows():
        src_path = Path(row.image_path)
        if not src_path.is_absolute():
            src_path = config.src_root / row.image_path
        if not src_path.exists():
            skipped.append(f"missing:{src_path}")
            continue
        try:
            with Image.open(src_path) as img:
                cropped = center_square_crop(img, config.ratio)
        except Exception as e:  # pragma: no cover
            skipped.append(f"fail:{src_path}:{e}")
            continue

        dst_path = out_img_dir / src_path.name
        cropped.save(dst_path)
        # Store path relative to the dataset root for portability
        rel_path = dst_path.relative_to(config.dst_root)
        rows.append((rel_path.as_posix(), row["class"]))

    if rows:
        labels_out = config.dst_root / config.labels_name
        with labels_out.open("w", encoding="utf-8") as f:
            f.write("image_path,class\n")
            for path, cls in rows:
                f.write(f"{path},{cls}\n")

    return len(rows), skipped


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Center crop UWF images to a narrower FOV.")
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root with labels.csv and images/")
    parser.add_argument("--dst", type=Path, required=True, help="Destination dataset root")
    parser.add_argument("--crop-deg", type=float, default=50.0, help="Target field of view in degrees")
    parser.add_argument("--full-deg", type=float, default=200.0, help="Assumed full FOV degrees of source")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = CropConfig(src_root=args.src, dst_root=args.dst, crop_deg=args.crop_deg, full_deg=args.full_deg)
    if not cfg.src_root.exists():
        print(f"Source root {cfg.src_root} does not exist; nothing to do.")
        return 0

    written, skipped = process(cfg)
    print(f"Cropped {written} images to {cfg.dst_root} (ratio={cfg.ratio:.3f}).")
    if skipped:
        print(f"Skipped {len(skipped)} items:")
        for s in skipped[:20]:
            print(f"  {s}")
        if len(skipped) > 20:
            print("  ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
