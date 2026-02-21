"""
Step 3: Feature Extraction for ML Baseline
============================================
Extracts features from fundus images (typically the 45° crops from Step 2)
for training a baseline classifier (Random Forest, SVM, Logistic Regression).

Feature groups:
  A. Color histograms     — per-channel histograms in RGB + HSV (~192 features)
  B. Regional color stats — mean/std/skew per channel in anatomical zones (~108 features)
  C. Anatomical features  — OD confidence, disc-macula geometry, asymmetry (~20 features)
  D. Texture features     — LBP-like stats, green-channel vessel proxy (~30 features)
  E. HOG features         — gradient histograms (~8k-20k features, saved separately)

Output:
  - features.csv          — Groups A-D, compact, human-readable
  - hog_features.npz      — Group E, high-dimensional arrays
  - feature_names.json    — Column name lists for both files
  - extraction_report.json — Summary stats, feature counts, any failures

Requirements:
    pip install opencv-python-headless numpy scikit-image pandas

Usage:
    # Extract from 45° crops (most common workflow)
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features

    # Use label mapping from directory structure (class_name/image.jpg)
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features --label_dir ./labels

    # Use a CSV label file
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features --label_csv labels.csv

    # Skip HOG (faster, if you only want compact features)
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features --skip_hog

    # Read pre-computed crop metadata from Step 2 (avoids re-detecting OD)
    python 3_extract_features.py --input_dir ./cropped_45deg --output_dir ./features --crop_meta ./cropped_45deg/crop_metadata.json
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

try:
    import pandas as pd
except ImportError:
    pd = None
    warnings.warn(
        "pandas not installed. CSV output will use manual writing. "
        "Install with: pip install pandas"
    )

try:
    from skimage.feature import hog as skimage_hog
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn(
        "scikit-image not installed. HOG and LBP features will be unavailable. "
        "Install with: pip install scikit-image"
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """All feature extraction parameters."""

    # --- Color histograms ---
    # Number of bins per channel for histograms.
    hist_bins: int = 32

    # --- Regional stats ---
    # Number of radial zones around the image center for regional analysis.
    # Zone 0 = central (foveal), Zone 1 = parafoveal, Zone 2 = peripapillary/peripheral
    num_radial_zones: int = 3

    # Quadrant analysis: split into 4 quadrants for asymmetry features.
    use_quadrants: bool = True

    # --- OD detection (reused from Step 2 for images without metadata) ---
    mask_threshold: int = 15
    morph_kernel_size: int = 15
    od_blur_ksize: int = 51
    od_channel: str = "red"
    od_diameter_fov_fraction: float = 0.05
    peripheral_exclusion_ratio: float = 0.25

    # --- HOG ---
    hog_orientations: int = 9
    hog_pixels_per_cell: Tuple[int, int] = (16, 16)
    hog_cells_per_block: Tuple[int, int] = (2, 2)
    hog_resize: int = 256  # Resize image before HOG to control vector length

    # --- LBP (texture) ---
    lbp_radius: int = 3
    lbp_n_points: int = 24  # Typically 8 * radius

    # --- Vessel proxy ---
    # Green-channel CLAHE + threshold for rough vessel density estimation.
    vessel_clahe_clip: float = 3.0
    vessel_clahe_grid: int = 8


# ──────────────────────────────────────────────
# OD Detection (minimal reuse from Step 2)
# ──────────────────────────────────────────────

def _build_fov_mask(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Binary mask separating retina from black background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, cfg.mask_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel_size, cfg.morph_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _build_central_mask(mask: np.ndarray, exclusion_ratio: float) -> np.ndarray:
    """Erode FOV mask to exclude outer periphery."""
    h, w = mask.shape[:2]
    erode_px = int(max(h, w) * exclusion_ratio)
    if erode_px < 3:
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    return cv2.erode(mask, kernel, iterations=1)


def _score_od_candidate(contour, fov_diameter, expected_frac):
    """Score a bright blob on OD likelihood."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or area == 0:
        return 0.0
    circularity = min(4 * np.pi * area / (perimeter ** 2 + 1e-6), 1.0)
    blob_d = np.sqrt(4 * area / np.pi)
    blob_frac = blob_d / (fov_diameter + 1e-6)
    size_score = np.exp(-0.5 * ((blob_frac - expected_frac) / 0.03) ** 2)
    if blob_frac > 0.15:
        size_score *= 0.1
    return circularity * 0.5 + size_score * 0.5


def detect_optic_disc(
    image: np.ndarray, mask: np.ndarray, cfg: FeatureConfig
) -> Tuple[int, int, float, float]:
    """
    Detect OD. Returns (cx, cy, confidence, detected_blob_area_fraction).
    """
    h, w = image.shape[:2]
    fov_diameter = max(h, w)
    central_mask = _build_central_mask(mask, cfg.peripheral_exclusion_ratio)

    channels = {"red": image[:, :, 2], "green": image[:, :, 1]}
    channel_order = [cfg.od_channel] + [c for c in channels if c != cfg.od_channel]

    best_cx, best_cy, best_score = w // 2, h // 2, 0.0
    best_area_frac = 0.0

    for ch_name in channel_order:
        channel = channels.get(ch_name)
        if channel is None:
            continue
        for search_mask, label in [(central_mask, "central"), (mask, "full")]:
            masked = cv2.bitwise_and(channel, channel, mask=search_mask)
            ksize = cfg.od_blur_ksize | 1
            blurred = cv2.GaussianBlur(masked, (ksize, ksize), 0)
            max_val = blurred.max()
            if max_val == 0:
                continue
            for frac in [0.92, 0.85, 0.75]:
                _, bright = cv2.threshold(blurred, int(max_val * frac), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    score = _score_od_candidate(cnt, fov_diameter, cfg.od_diameter_fov_fraction)
                    if label == "central":
                        score *= 1.3
                    if ch_name == cfg.od_channel:
                        score *= 1.1
                    if score > best_score:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            best_cx = int(M["m10"] / M["m00"])
                            best_cy = int(M["m01"] / M["m00"])
                            best_score = score
                            area = cv2.contourArea(cnt)
                            best_area_frac = area / (fov_diameter ** 2 + 1e-6)
                if best_score > 0.4:
                    break
            if label == "central" and best_score > 0.3:
                break
        if best_score > 0.5:
            break

    return best_cx, best_cy, min(best_score, 1.0), best_area_frac


# ──────────────────────────────────────────────
# Feature Group A: Color Histograms
# ──────────────────────────────────────────────

def extract_color_histograms(
    image: np.ndarray, mask: np.ndarray, cfg: FeatureConfig
) -> Dict[str, float]:
    """
    Per-channel histograms in BGR and HSV color spaces.
    Histograms are normalized (sum=1) so they're scale-invariant.
    """
    features = {}
    bins = cfg.hist_bins

    # BGR histograms
    for i, name in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([image], [i], mask, [bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        for b in range(bins):
            features[f"hist_bgr_{name}_bin{b:02d}"] = float(hist[b])

    # HSV histograms
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i, (name, upper) in enumerate([("hue", 180), ("sat", 256), ("val", 256)]):
        hist = cv2.calcHist([hsv], [i], mask, [bins], [0, upper])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        for b in range(bins):
            features[f"hist_hsv_{name}_bin{b:02d}"] = float(hist[b])

    return features


# ──────────────────────────────────────────────
# Feature Group B: Regional Color Statistics
# ──────────────────────────────────────────────

def _compute_channel_stats(
    channel: np.ndarray, zone_mask: np.ndarray, prefix: str
) -> Dict[str, float]:
    """Mean, std, skewness, and percentiles for a channel within a masked zone."""
    pixels = channel[zone_mask > 0].astype(np.float64)
    features = {}

    if len(pixels) == 0:
        for suffix in ["mean", "std", "skew", "p10", "p50", "p90"]:
            features[f"{prefix}_{suffix}"] = 0.0
        return features

    mean = np.mean(pixels)
    std = np.std(pixels)
    features[f"{prefix}_mean"] = float(mean)
    features[f"{prefix}_std"] = float(std)

    # Skewness (Fisher's definition)
    if std > 1e-8:
        features[f"{prefix}_skew"] = float(np.mean(((pixels - mean) / std) ** 3))
    else:
        features[f"{prefix}_skew"] = 0.0

    features[f"{prefix}_p10"] = float(np.percentile(pixels, 10))
    features[f"{prefix}_p50"] = float(np.percentile(pixels, 50))
    features[f"{prefix}_p90"] = float(np.percentile(pixels, 90))

    return features


def extract_regional_color_stats(
    image: np.ndarray, mask: np.ndarray, cfg: FeatureConfig
) -> Dict[str, float]:
    """
    Color statistics computed in concentric radial zones and quadrants.

    Radial zones (from center outward):
      zone_0 = central/foveal (inner third)
      zone_1 = parafoveal (middle third)
      zone_2 = peripapillary/peripheral (outer third)

    Quadrants:
      q0 = top-left, q1 = top-right, q2 = bottom-left, q3 = bottom-right

    Computed per zone: mean, std, skewness, p10, p50, p90 for each of
    R, G, B, H, S, V channels.
    """
    features = {}
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    max_radius = np.sqrt(cx ** 2 + cy ** 2)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build distance map from center
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # --- Radial zones ---
    for zone_idx in range(cfg.num_radial_zones):
        r_inner = (zone_idx / cfg.num_radial_zones) * max_radius
        r_outer = ((zone_idx + 1) / cfg.num_radial_zones) * max_radius

        zone_mask = ((dist >= r_inner) & (dist < r_outer)).astype(np.uint8) * 255
        zone_mask = cv2.bitwise_and(zone_mask, mask)

        for i, name in enumerate(["blue", "green", "red"]):
            prefix = f"zone{zone_idx}_{name}"
            features.update(_compute_channel_stats(image[:, :, i], zone_mask, prefix))

        for i, name in enumerate(["hue", "sat", "val"]):
            prefix = f"zone{zone_idx}_{name}"
            features.update(_compute_channel_stats(hsv[:, :, i], zone_mask, prefix))

    # --- Quadrant analysis ---
    if cfg.use_quadrants:
        for qi, (y_slice, x_slice, qname) in enumerate([
            (slice(0, cy), slice(0, cx), "q_topleft"),
            (slice(0, cy), slice(cx, w), "q_topright"),
            (slice(cy, h), slice(0, cx), "q_bottomleft"),
            (slice(cy, h), slice(cx, w), "q_bottomright"),
        ]):
            q_mask = np.zeros((h, w), dtype=np.uint8)
            q_mask[y_slice, x_slice] = 255
            q_mask = cv2.bitwise_and(q_mask, mask)

            for i, name in enumerate(["red", "green", "blue"]):
                prefix = f"{qname}_{name}"
                features.update(_compute_channel_stats(image[:, :, 2 - i], q_mask, prefix))

        # Asymmetry features: left-right and top-bottom differences
        for name_idx, name in enumerate(["red", "green", "blue"]):
            ch = 2 - name_idx
            left_pixels = image[:, :cx, ch][mask[:, :cx] > 0].astype(np.float64)
            right_pixels = image[:, cx:, ch][mask[:, cx:] > 0].astype(np.float64)
            top_pixels = image[:cy, :, ch][mask[:cy, :] > 0].astype(np.float64)
            bottom_pixels = image[cy:, :, ch][mask[cy:, :] > 0].astype(np.float64)

            l_mean = np.mean(left_pixels) if len(left_pixels) > 0 else 0
            r_mean = np.mean(right_pixels) if len(right_pixels) > 0 else 0
            t_mean = np.mean(top_pixels) if len(top_pixels) > 0 else 0
            b_mean = np.mean(bottom_pixels) if len(bottom_pixels) > 0 else 0

            features[f"asymmetry_lr_{name}"] = float(abs(l_mean - r_mean))
            features[f"asymmetry_tb_{name}"] = float(abs(t_mean - b_mean))
            denom = l_mean + r_mean + 1e-8
            features[f"asymmetry_lr_{name}_ratio"] = float(abs(l_mean - r_mean) / denom)
            denom = t_mean + b_mean + 1e-8
            features[f"asymmetry_tb_{name}_ratio"] = float(abs(t_mean - b_mean) / denom)

    return features


# ──────────────────────────────────────────────
# Feature Group C: Anatomical Features
# ──────────────────────────────────────────────

def extract_anatomical_features(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: FeatureConfig,
    precomputed_od: Optional[Tuple[int, int, float]] = None,
) -> Dict[str, float]:
    """
    Features derived from OD detection and peripapillary analysis.

    Uses pre-computed OD coordinates if available (from crop_metadata.json),
    otherwise detects OD from scratch.
    """
    features = {}
    h, w = image.shape[:2]
    fov_diameter = max(h, w)

    # Get OD location
    if precomputed_od is not None:
        od_x, od_y, od_conf = precomputed_od
        od_area_frac = 0.0  # Not available from precomputed
    else:
        od_x, od_y, od_conf, od_area_frac = detect_optic_disc(image, mask, cfg)

    features["od_x_normalized"] = float(od_x / (w + 1e-8))
    features["od_y_normalized"] = float(od_y / (h + 1e-8))
    features["od_confidence"] = float(od_conf)
    features["od_detected_area_frac"] = float(od_area_frac)

    # Distance from OD to image center (normalized)
    cx, cy = w // 2, h // 2
    dist_to_center = np.sqrt((od_x - cx) ** 2 + (od_y - cy) ** 2)
    features["od_dist_to_center_norm"] = float(dist_to_center / (fov_diameter / 2 + 1e-8))

    # OD region color analysis
    # Extract a circular patch around the detected OD
    dd_px = int(fov_diameter * cfg.od_diameter_fov_fraction)
    od_radius = max(dd_px // 2, 10)

    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, (od_x, od_y), od_radius, 255, -1)
    od_mask = cv2.bitwise_and(od_mask, mask)

    # Peripapillary ring: annulus around OD (1x to 2x OD radius)
    peri_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(peri_mask, (od_x, od_y), od_radius * 2, 255, -1)
    peri_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(peri_inner, (od_x, od_y), od_radius, 255, -1)
    peri_mask = cv2.subtract(peri_mask, peri_inner)
    peri_mask = cv2.bitwise_and(peri_mask, mask)

    # Background region: everything outside 3x OD radius
    bg_mask = mask.copy()
    bg_exclude = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bg_exclude, (od_x, od_y), od_radius * 3, 255, -1)
    bg_mask = cv2.subtract(bg_mask, bg_exclude)

    for i, name in enumerate(["blue", "green", "red"]):
        ch = image[:, :, i]

        od_pixels = ch[od_mask > 0].astype(np.float64)
        peri_pixels = ch[peri_mask > 0].astype(np.float64)
        bg_pixels = ch[bg_mask > 0].astype(np.float64)

        od_mean = np.mean(od_pixels) if len(od_pixels) > 0 else 0
        peri_mean = np.mean(peri_pixels) if len(peri_pixels) > 0 else 0
        bg_mean = np.mean(bg_pixels) if len(bg_pixels) > 0 else 0

        features[f"od_region_{name}_mean"] = float(od_mean)
        features[f"peri_region_{name}_mean"] = float(peri_mean)

        # OD-to-background contrast (disc pallor proxy)
        features[f"od_bg_contrast_{name}"] = float(od_mean - bg_mean)

        # Peripapillary-to-background ratio (PPA proxy)
        features[f"peri_bg_ratio_{name}"] = float(
            peri_mean / (bg_mean + 1e-8)
        )

    # Peripapillary asymmetry (4 quadrants around the disc)
    for qi, angle_range in enumerate([(0, 90), (90, 180), (180, 270), (270, 360)]):
        q_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            q_mask, (od_x, od_y), (od_radius * 2, od_radius * 2),
            0, angle_range[0], angle_range[1], 255, -1
        )
        q_mask = cv2.bitwise_and(q_mask, peri_mask)

        red_pixels = image[:, :, 2][q_mask > 0].astype(np.float64)
        features[f"peri_quadrant{qi}_red_mean"] = float(
            np.mean(red_pixels) if len(red_pixels) > 0 else 0
        )

    # Green-to-red ratio in central vs peripheral zones
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_radius = int(fov_diameter * 0.15)
    cv2.circle(center_mask, (cx, cy), center_radius, 255, -1)
    center_mask = cv2.bitwise_and(center_mask, mask)

    center_green = image[:, :, 1][center_mask > 0].astype(np.float64)
    center_red = image[:, :, 2][center_mask > 0].astype(np.float64)
    features["center_green_red_ratio"] = float(
        np.mean(center_green) / (np.mean(center_red) + 1e-8)
    ) if len(center_green) > 0 else 0.0

    return features


# ──────────────────────────────────────────────
# Feature Group D: Texture Features
# ──────────────────────────────────────────────

def extract_texture_features(
    image: np.ndarray, mask: np.ndarray, cfg: FeatureConfig
) -> Dict[str, float]:
    """
    Texture features: LBP histogram + green-channel vessel density proxy.
    """
    features = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- LBP ---
    if HAS_SKIMAGE:
        lbp = local_binary_pattern(
            gray, cfg.lbp_n_points, cfg.lbp_radius, method="uniform"
        )
        # Uniform LBP has n_points + 2 possible values
        n_bins = cfg.lbp_n_points + 2
        lbp_masked = lbp[mask > 0]

        if len(lbp_masked) > 0:
            hist, _ = np.histogram(
                lbp_masked, bins=n_bins, range=(0, n_bins), density=True
            )
            for b in range(n_bins):
                features[f"lbp_bin{b:02d}"] = float(hist[b])

            features["lbp_entropy"] = float(
                -np.sum(hist * np.log2(hist + 1e-10))
            )
        else:
            for b in range(n_bins):
                features[f"lbp_bin{b:02d}"] = 0.0
            features["lbp_entropy"] = 0.0

    # --- Green-channel vessel density proxy ---
    green = image[:, :, 1]

    # CLAHE for local contrast enhancement (makes vessels pop)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.vessel_clahe_clip,
        tileGridSize=(cfg.vessel_clahe_grid, cfg.vessel_clahe_grid),
    )
    enhanced = clahe.apply(green)

    # Invert (vessels are dark in green channel → become bright)
    enhanced_inv = cv2.bitwise_not(enhanced)
    enhanced_inv = cv2.bitwise_and(enhanced_inv, enhanced_inv, mask=mask)

    # Adaptive threshold for vessel-like structures
    vessel_binary = cv2.adaptiveThreshold(
        enhanced_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    vessel_binary = cv2.bitwise_and(vessel_binary, mask)

    # Vessel density = fraction of FOV pixels that are "vessel"
    fov_pixels = cv2.countNonZero(mask)
    vessel_pixels = cv2.countNonZero(vessel_binary)
    features["vessel_density"] = float(
        vessel_pixels / (fov_pixels + 1e-8)
    )

    # Vessel density in center vs periphery
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_r = int(max(h, w) * 0.25)
    cv2.circle(center_mask, (w // 2, h // 2), center_r, 255, -1)
    center_mask = cv2.bitwise_and(center_mask, mask)

    periph_mask = cv2.subtract(mask, center_mask)

    center_vessel = cv2.countNonZero(cv2.bitwise_and(vessel_binary, center_mask))
    center_total = cv2.countNonZero(center_mask)
    periph_vessel = cv2.countNonZero(cv2.bitwise_and(vessel_binary, periph_mask))
    periph_total = cv2.countNonZero(periph_mask)

    features["vessel_density_center"] = float(center_vessel / (center_total + 1e-8))
    features["vessel_density_periph"] = float(periph_vessel / (periph_total + 1e-8))
    features["vessel_density_ratio_cp"] = float(
        (center_vessel / (center_total + 1e-8)) /
        (periph_vessel / (periph_total + 1e-8) + 1e-8)
    )

    # Green channel overall stats (complementary to color histograms)
    green_masked = green[mask > 0].astype(np.float64)
    if len(green_masked) > 0:
        features["green_channel_mean"] = float(np.mean(green_masked))
        features["green_channel_std"] = float(np.std(green_masked))
        features["green_channel_entropy"] = float(
            -np.sum(
                np.histogram(green_masked, bins=64, range=(0, 256), density=True)[0]
                * np.log2(
                    np.histogram(green_masked, bins=64, range=(0, 256), density=True)[0] + 1e-10
                )
            )
        )
    else:
        features["green_channel_mean"] = 0.0
        features["green_channel_std"] = 0.0
        features["green_channel_entropy"] = 0.0

    return features


# ──────────────────────────────────────────────
# Feature Group E: HOG
# ──────────────────────────────────────────────

def extract_hog_features(
    image: np.ndarray, cfg: FeatureConfig
) -> Optional[np.ndarray]:
    """
    HOG descriptor computed on a resized grayscale version of the image.
    Returns a 1D numpy array, or None if scikit-image is unavailable.
    """
    if not HAS_SKIMAGE:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (cfg.hog_resize, cfg.hog_resize))

    hog_vector = skimage_hog(
        resized,
        orientations=cfg.hog_orientations,
        pixels_per_cell=cfg.hog_pixels_per_cell,
        cells_per_block=cfg.hog_cells_per_block,
        feature_vector=True,
    )

    return hog_vector.astype(np.float32)


# ──────────────────────────────────────────────
# Label Loading
# ──────────────────────────────────────────────

def load_labels_from_csv(csv_path: str) -> Dict[str, str]:
    """
    Load labels from a CSV file. Expected columns: filename, label.
    Also accepts: image, class or file, category (flexible matching).
    """
    labels = {}
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        header_lower = [h.strip().lower() for h in header]

        # Find filename column
        file_col = None
        for candidate in ["filename", "file", "image", "image_name", "name"]:
            if candidate in header_lower:
                file_col = header_lower.index(candidate)
                break
        if file_col is None:
            file_col = 0

        # Find label column
        label_col = None
        for candidate in ["label", "class", "category", "diagnosis", "target"]:
            if candidate in header_lower:
                label_col = header_lower.index(candidate)
                break
        if label_col is None:
            label_col = 1

        for line in f:
            parts = line.strip().split(",")
            if len(parts) > max(file_col, label_col):
                fname = parts[file_col].strip().strip('"')
                label = parts[label_col].strip().strip('"')
                labels[fname] = label

    return labels


def load_labels_from_dir(label_dir: str) -> Dict[str, str]:
    """
    Load labels from a directory structure: label_dir/class_name/image.jpg
    """
    labels = {}
    label_path = Path(label_dir)
    for class_dir in sorted(label_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                labels[img_file.name] = class_name
    return labels


# ──────────────────────────────────────────────
# Main Extraction Pipeline
# ──────────────────────────────────────────────

def extract_all_features(
    input_dir: str,
    output_dir: str,
    cfg: FeatureConfig,
    labels: Optional[Dict[str, str]] = None,
    crop_metadata: Optional[Dict[str, dict]] = None,
    skip_hog: bool = False,
) -> None:
    """
    Extract features from all images in input_dir and save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        print(f"No images found in {input_dir}")
        return

    all_compact_features = []
    all_hog_vectors = []
    all_filenames = []
    compact_feature_names = None
    failures = []

    for i, fpath in enumerate(files, 1):
        filename = fpath.name
        print(f"[{i}/{len(files)}] {filename}...", end=" ", flush=True)

        try:
            image = cv2.imread(str(fpath))
            if image is None:
                raise FileNotFoundError(f"Could not read: {fpath}")

            mask = _build_fov_mask(image, cfg)

            # Get supplementary info from Step 2 metadata (if available)
            meta_laterality = None
            meta_od_conf_original = None
            if crop_metadata and filename in crop_metadata:
                meta = crop_metadata[filename]
                if meta.get("success", False):
                    meta_laterality = meta.get("laterality", None)
                    meta_od_conf_original = meta.get("od_confidence", None)

            # Extract feature groups
            features = {}

            # A: Color histograms
            features.update(extract_color_histograms(image, mask, cfg))

            # B: Regional color stats
            features.update(extract_regional_color_stats(image, mask, cfg))

            # C: Anatomical features (always re-detects OD on the input image)
            features.update(
                extract_anatomical_features(image, mask, cfg, precomputed_od=None)
            )

            # Add supplementary metadata features from Step 2
            if meta_laterality:
                features["laterality_is_od"] = 1.0 if meta_laterality == "OD" else 0.0
            if meta_od_conf_original is not None:
                features["od_confidence_fullimg"] = float(meta_od_conf_original)

            # D: Texture features
            features.update(extract_texture_features(image, mask, cfg))

            # Add label if available
            if labels and filename in labels:
                features["label"] = labels[filename]

            all_compact_features.append(features)
            all_filenames.append(filename)

            if compact_feature_names is None:
                compact_feature_names = [
                    k for k in features.keys() if k != "label"
                ]

            # E: HOG
            if not skip_hog:
                hog_vec = extract_hog_features(image, cfg)
                if hog_vec is not None:
                    all_hog_vectors.append(hog_vec)

            print("OK")

        except Exception as e:
            print(f"FAIL — {e}")
            failures.append({"filename": filename, "error": str(e)})

    # ── Save compact features ──
    csv_path = os.path.join(output_dir, "features.csv")

    if pd is not None:
        df = pd.DataFrame(all_compact_features)
        df.insert(0, "filename", all_filenames)
        df.to_csv(csv_path, index=False)
    else:
        # Manual CSV writing
        if all_compact_features:
            all_keys = ["filename"] + compact_feature_names
            has_labels = "label" in all_compact_features[0]
            if has_labels:
                all_keys.append("label")

            with open(csv_path, "w") as f:
                f.write(",".join(all_keys) + "\n")
                for fname, feat in zip(all_filenames, all_compact_features):
                    row = [fname]
                    for k in compact_feature_names:
                        row.append(str(feat.get(k, 0)))
                    if has_labels:
                        row.append(feat.get("label", ""))
                    f.write(",".join(row) + "\n")

    print(f"\nCompact features saved to {csv_path}")

    # ── Save HOG features ──
    if all_hog_vectors and not skip_hog:
        hog_path = os.path.join(output_dir, "hog_features.npz")
        hog_matrix = np.array(all_hog_vectors)
        np.savez_compressed(
            hog_path,
            features=hog_matrix,
            filenames=np.array(all_filenames[:len(all_hog_vectors)]),
        )
        print(f"HOG features saved to {hog_path}  (shape: {hog_matrix.shape})")

    # ── Save feature name registry ──
    names_path = os.path.join(output_dir, "feature_names.json")
    name_registry = {
        "compact_features": compact_feature_names or [],
        "compact_feature_count": len(compact_feature_names or []),
        "hog_vector_length": len(all_hog_vectors[0]) if all_hog_vectors else 0,
        "total_feature_count": (
            len(compact_feature_names or [])
            + (len(all_hog_vectors[0]) if all_hog_vectors else 0)
        ),
    }
    with open(names_path, "w") as f:
        json.dump(name_registry, f, indent=2)

    # ── Save extraction report ──
    report_path = os.path.join(output_dir, "extraction_report.json")
    report = {
        "total_images": len(files),
        "successful": len(all_filenames),
        "failed": len(failures),
        "compact_feature_count": len(compact_feature_names or []),
        "hog_vector_length": len(all_hog_vectors[0]) if all_hog_vectors else 0,
        "hog_skipped": skip_hog,
        "labels_provided": labels is not None,
        "unique_labels": sorted(set(
            f.get("label", "") for f in all_compact_features if "label" in f
        )) if labels else [],
        "label_counts": {},
        "failures": failures,
    }
    if labels:
        for feat in all_compact_features:
            lbl = feat.get("label", "unknown")
            report["label_counts"][lbl] = report["label_counts"].get(lbl, 0) + 1

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Feature names saved to {names_path}")
    print(f"Report saved to {report_path}")
    print(f"\nDone: {len(all_filenames)} images, "
          f"{len(compact_feature_names or [])} compact features"
          + (f" + {len(all_hog_vectors[0])} HOG features" if all_hog_vectors else "")
          + f" per image.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Extract ML features from fundus images"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of images (typically cropped_45deg from Step 2)")
    parser.add_argument("--output_dir", type=str, default="./features",
                        help="Output directory for feature files (default: ./features)")

    # Labels
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--label_csv", type=str,
                             help="CSV file with columns: filename, label")
    label_group.add_argument("--label_dir", type=str,
                             help="Directory with class_name/image.jpg structure")

    # Pre-computed metadata
    parser.add_argument("--crop_meta", type=str,
                        help="Path to crop_metadata.json from Step 2. "
                             "Used for OD confidence as a feature. "
                             "OD position is always re-detected on the input images "
                             "since crop coordinates differ from standardized coordinates.")

    # Feature toggles
    parser.add_argument("--skip_hog", action="store_true",
                        help="Skip HOG feature extraction (faster)")
    parser.add_argument("--hist_bins", type=int, default=32,
                        help="Histogram bins per channel (default: 32)")
    parser.add_argument("--hog_resize", type=int, default=256,
                        help="Resize dimension for HOG (default: 256)")

    args = parser.parse_args()

    cfg = FeatureConfig(
        hist_bins=args.hist_bins,
        hog_resize=args.hog_resize,
    )

    # Load labels
    labels = None
    if args.label_csv:
        labels = load_labels_from_csv(args.label_csv)
        print(f"Loaded {len(labels)} labels from {args.label_csv}")
    elif args.label_dir:
        labels = load_labels_from_dir(args.label_dir)
        print(f"Loaded {len(labels)} labels from {args.label_dir}")

    # Load crop metadata
    crop_meta = None
    if args.crop_meta:
        with open(args.crop_meta) as f:
            meta_raw = json.load(f)
        # Handle both formats: list of results or dict with "results" key
        results_list = meta_raw if isinstance(meta_raw, list) else meta_raw.get("results", [])
        crop_meta = {r["filename"]: r for r in results_list}
        print(f"Loaded crop metadata for {len(crop_meta)} images")

    extract_all_features(
        args.input_dir, args.output_dir, cfg,
        labels=labels,
        crop_metadata=crop_meta,
        skip_hog=args.skip_hog,
    )


if __name__ == "__main__":
    main()