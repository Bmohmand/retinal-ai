"""
Step 3: Feature Extraction for SVM Baseline
=============================================
Extracts HOG descriptors and color intensity histograms from fundus images
for training an SVM baseline classifier (as described in the project proposal).

Feature groups:
  A. Color intensity histograms — per-channel in RGB + HSV  (192 features)
  B. HOG descriptors           — gradient texture patterns   (~8,100 features)

Run this TWICE: once on the full 200° standardized images, once on the 45°
crops. Then train identical SVMs on each to compare field-of-view performance.

Output:
  - features.csv      — Color histograms (compact, human-readable)
  - hog_features.npz  — HOG vectors (high-dimensional, compressed)
  - extraction_report.json — Summary stats

Requirements:
    pip install opencv-python-headless numpy scikit-image pandas

Usage:
    # Extract from full 200° standardized images
    python 3_extract_features.py --input_dir ./standardized --output_dir ./features_200deg --label_csv labels.csv

    # Extract from 45° synthetic crops
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features_45deg --label_csv labels.csv

    # Skip HOG (faster iteration on color features only)
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features_45deg --label_csv labels.csv --skip_hog
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict

try:
    import pandas as pd
except ImportError:
    pd = None
    warnings.warn("pandas not installed. Install with: pip install pandas")

try:
    from skimage.feature import hog as skimage_hog
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn(
        "scikit-image not installed. HOG will be unavailable. "
        "Install with: pip install scikit-image"
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Color histograms
HIST_BINS = 32          # Bins per channel (32 × 6 channels = 192 features)

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_RESIZE = 256        # Resize before HOG to standardize vector length

# Mask threshold for separating retina from black background
MASK_THRESHOLD = 15
MORPH_KERNEL = 15


# ──────────────────────────────────────────────
# FOV Mask
# ──────────────────────────────────────────────

def build_fov_mask(image: np.ndarray) -> np.ndarray:
    """
    Binary mask of retinal pixels (excludes black corners/background).
    Used to compute histograms only over actual retinal content.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


# ──────────────────────────────────────────────
# Feature Group A: Color Intensity Histograms
# ──────────────────────────────────────────────

def extract_color_histograms(
    image: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    """
    Per-channel normalized histograms in BGR and HSV color spaces.

    Captures pigmentation variance across tumor types:
      - Melanoma: dark brown/gray intensity profile
      - Retinoblastoma: bright white/yellow peak
      - Hemangioma: strong red-channel skew
      - Normal: uniform red-orange distribution

    Returns dict of 192 features (32 bins × 6 channels).
    """
    features = {}

    # BGR histograms
    for i, name in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([image], [i], mask, [HIST_BINS], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        for b in range(HIST_BINS):
            features[f"hist_{name}_bin{b:02d}"] = float(hist[b])

    # HSV histograms
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i, (name, upper) in enumerate([("hue", 180), ("sat", 256), ("val", 256)]):
        hist = cv2.calcHist([hsv], [i], mask, [HIST_BINS], [0, upper])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        for b in range(HIST_BINS):
            features[f"hist_{name}_bin{b:02d}"] = float(hist[b])

    return features


# ──────────────────────────────────────────────
# Feature Group B: HOG Descriptors
# ──────────────────────────────────────────────

def extract_hog_features(image: np.ndarray) -> Optional[np.ndarray]:
    """
    HOG descriptor on resized grayscale image.

    Captures texture and edge patterns — tumor boundaries, vessel
    architecture, calcification edges.

    Returns 1D float32 array (~8,100 features at 256x256 with default params).
    """
    if not HAS_SKIMAGE:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HOG_RESIZE, HOG_RESIZE))

    hog_vector = skimage_hog(
        resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )

    return hog_vector.astype(np.float32)


# ──────────────────────────────────────────────
# Label Loading
# ──────────────────────────────────────────────

def load_labels_from_csv(csv_path: str) -> Dict[str, str]:
    """
    Load labels from CSV. Flexibly matches column names.
    Expected: filename/image column + label/class/diagnosis column.
    """
    labels = {}
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        header_lower = [h.strip().lower().strip('"') for h in header]

        file_col = 0
        for candidate in ["filename", "file", "image", "image_name", "name"]:
            if candidate in header_lower:
                file_col = header_lower.index(candidate)
                break

        label_col = 1
        for candidate in ["label", "class", "category", "diagnosis", "target"]:
            if candidate in header_lower:
                label_col = header_lower.index(candidate)
                break

        for line in f:
            parts = line.strip().split(",")
            if len(parts) > max(file_col, label_col):
                fname = parts[file_col].strip().strip('"')
                label = parts[label_col].strip().strip('"')
                labels[fname] = label

    return labels


def load_labels_from_dir(label_dir: str) -> Dict[str, str]:
    """Load labels from directory structure: label_dir/class_name/image.jpg"""
    labels = {}
    for class_dir in sorted(Path(label_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        for img in sorted(class_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                labels[img.name] = class_dir.name
    return labels


# ──────────────────────────────────────────────
# Main Extraction
# ──────────────────────────────────────────────

def extract_all(
    input_dir: str,
    output_dir: str,
    labels: Optional[Dict[str, str]] = None,
    skip_hog: bool = False,
):
    """Extract features from all images and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)

    print(f"Scanning for images in {input_dir}...")
    files = sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        print(f"No images found in {input_dir}")
        return

    all_features = []
    all_hog = []
    all_filenames = []
    feature_names = None
    failures = []

    for i, fpath in enumerate(files, 1):
        rel_path = fpath.relative_to(input_path)
        filename = fpath.name
        print(f"[{i}/{len(files)}] {rel_path}...", end=" ", flush=True)

        try:
            image = cv2.imread(str(fpath))
            if image is None:
                raise FileNotFoundError(f"Could not read: {fpath}")

            mask = build_fov_mask(image)

            # Color histograms
            features = extract_color_histograms(image, mask)

            # Label
            if labels:
                # Try relative path first, then basename
                lbl = labels.get(str(rel_path)) or labels.get(filename)
                if lbl:
                    features["label"] = lbl

            all_features.append(features)
            all_filenames.append(str(rel_path))

            if feature_names is None:
                feature_names = [k for k in features if k != "label"]

            # HOG
            if not skip_hog:
                hog_vec = extract_hog_features(image)
                if hog_vec is not None:
                    all_hog.append(hog_vec)

            print("OK")

        except Exception as e:
            print(f"FAIL — {e}")
            failures.append({"filename": str(rel_path), "error": str(e)})

    # ── Save color histograms to CSV ──
    csv_path = os.path.join(output_dir, "features.csv")
    if pd is not None:
        df = pd.DataFrame(all_features)
        df.insert(0, "filename", all_filenames)
        df.to_csv(csv_path, index=False)
    else:
        has_labels = labels is not None
        cols = ["filename"] + feature_names + (["label"] if has_labels else [])
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            for fname, feat in zip(all_filenames, all_features):
                row = [fname] + [str(feat.get(k, 0)) for k in feature_names]
                if has_labels:
                    row.append(feat.get("label", ""))
                f.write(",".join(row) + "\n")

    print(f"\nColor histograms saved to {csv_path}  ({len(feature_names)} features)")

    # ── Save HOG to .npz ──
    if all_hog and not skip_hog:
        hog_path = os.path.join(output_dir, "hog_features.npz")
        hog_matrix = np.array(all_hog)
        np.savez_compressed(
            hog_path,
            features=hog_matrix,
            filenames=np.array(all_filenames[:len(all_hog)]),
        )
        print(f"HOG features saved to {hog_path}  ({hog_matrix.shape[1]} features, shape: {hog_matrix.shape})")

    # ── Save report ──
    report = {
        "input_dir": input_dir,
        "total_images": len(files),
        "successful": len(all_filenames),
        "failed": len(failures),
        "color_histogram_features": len(feature_names) if feature_names else 0,
        "hog_vector_length": len(all_hog[0]) if all_hog else 0,
        "hog_skipped": skip_hog,
        "hist_bins_per_channel": HIST_BINS,
        "hog_resize": HOG_RESIZE,
        "labels_provided": labels is not None,
        "unique_labels": sorted(set(
            f.get("label", "") for f in all_features if "label" in f
        )) if labels else [],
        "label_counts": {},
        "failures": failures,
    }
    if labels:
        for feat in all_features:
            lbl = feat.get("label", "unknown")
            report["label_counts"][lbl] = report["label_counts"].get(lbl, 0) + 1

    report_path = os.path.join(output_dir, "extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")
    print(f"\nDone: {len(all_filenames)} images — "
          f"{len(feature_names or [])} color features"
          + (f" + {len(all_hog[0])} HOG features" if all_hog else "")
          + " per image.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Extract HOG + color histogram features for SVM baseline"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Image directory (standardized or cropped_45deg)")
    parser.add_argument("--output_dir", type=str, default="./features",
                        help="Output directory (default: ./features)")

    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--label_csv", type=str,
                             help="CSV with filename + label columns")
    label_group.add_argument("--label_dir", type=str,
                             help="Directory with class_name/image.jpg structure")

    parser.add_argument("--skip_hog", action="store_true",
                        help="Skip HOG extraction (faster)")
    parser.add_argument("--hist_bins", type=int, default=32,
                        help="Bins per histogram channel (default: 32)")
    parser.add_argument("--hog_resize", type=int, default=256,
                        help="Image size for HOG computation (default: 256)")

    args = parser.parse_args()

    global HIST_BINS, HOG_RESIZE
    HIST_BINS = args.hist_bins
    HOG_RESIZE = args.hog_resize

    labels = None
    if args.label_csv:
        labels = load_labels_from_csv(args.label_csv)
        print(f"Loaded {len(labels)} labels from {args.label_csv}")
    elif args.label_dir:
        labels = load_labels_from_dir(args.label_dir)
        print(f"Loaded {len(labels)} labels from directory structure")

    extract_all(args.input_dir, args.output_dir, labels, args.skip_hog)


if __name__ == "__main__":
    main()