"""Feature extraction: HOG + color histograms."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog


def load_image(path: Path, image_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image missing: {path}")
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.array(img)


def extract_hog_color(path: Path, image_size: int = 256) -> np.ndarray:
    arr = load_image(path, image_size)
    gray = rgb2gray(arr)
    hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    # Simple color histogram (normalized) across channels
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[..., c], bins=32, range=(0, 255), density=True)
        hist.append(h)
    hist_feat = np.concatenate(hist)
    return np.concatenate([hog_feat, hist_feat]).astype(np.float32)
