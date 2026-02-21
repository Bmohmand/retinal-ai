"""Generate tiny synthetic retinal-like mock images for smoke tests."""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = [
    "choroidal_melanoma",
    "retinoblastoma",
    "choroidal_nevus",
    "hemangioma",
    "metastasis",
    "healthy",
]


def make_circle_image(size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Simulate a fundus-like gradient
    center = (size // 2 + rng.randint(-5, 5), size // 2 + rng.randint(-5, 5))
    radius = size // 2 - 4
    base_color = (rng.randint(120, 200), rng.randint(60, 140), rng.randint(40, 110))
    for r in range(radius, 0, -1):
        shade = tuple(int(c * (0.6 + 0.4 * r / radius)) for c in base_color)
        draw.ellipse(
            [center[0] - r, center[1] - r, center[0] + r, center[1] + r],
            fill=shade,
        )
    # Add a few bright spots to mimic lesions
    for _ in range(rng.randint(1, 4)):
        rr = rng.randint(4, 10)
        cx = rng.randint(rr, size - rr)
        cy = rng.randint(rr, size - rr)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(230, 230, 120))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Output directory for images and labels.csv")
    parser.add_argument("--count", type=int, default=24, help="Number of samples to create")
    parser.add_argument("--size", type=int, default=256, help="Image side length in pixels")
    args = parser.parse_args()

    out = args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(args.count):
        cls = CLASSES[idx % len(CLASSES)]
        seed = idx * 17 + 11
        img = make_circle_image(args.size, seed)
        fname = f"mock_{idx:03d}.png"
        img_path = img_dir / fname
        img.save(img_path)
        rel_path = img_path.as_posix()
        rows.append((rel_path, cls))

    labels_path = out / "labels.csv"
    with labels_path.open("w", encoding="utf-8") as f:
        f.write("image_path,class\n")
        for path, cls in rows:
            f.write(f"{path},{cls}\n")

    print(f"Wrote {len(rows)} samples to {out}")


if __name__ == "__main__":
    main()
