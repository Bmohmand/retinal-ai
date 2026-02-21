"""
Step 2: Synthetic 45° Crop from Standardized UWF Images
========================================================
Takes preprocessed/standardized UWF fundus images (output of 1_preprocess.py)
and produces synthetic 45-degree field-of-view crops, simulating a standard
fundus camera capture.

Pipeline per image:
  1. Build FOV mask (for the standardized image)
  2. Detect the optic disc (with peripheral exclusion + candidate scoring)
  3. Estimate the macula position (temporal to the disc)
  4. Extract a square crop representing the target FOV

Output: A directory of 45° crops + metadata JSON with detection coordinates.

Requirements:
    pip install opencv-python-headless numpy

Usage:
    # Standard workflow (run after 1_preprocess.py)
    python 2_synthetic_crop.py --input_dir ./standardized --output_dir ./cropped_45deg

    # With debug overlays to QA optic disc detection
    python 2_synthetic_crop.py --input_dir ./standardized --output_dir ./cropped_45deg --debug

    # Center on the optic disc instead of estimated macula
    python 2_synthetic_crop.py --input_dir ./standardized --output_dir ./cropped_45deg --center_on optic_disc

    # For Clarus images (133° FOV instead of Optos 200°)
    python 2_synthetic_crop.py --input_dir ./standardized --output_dir ./cropped_45deg --uwf_fov 133

    # Single image
    python 2_synthetic_crop.py --input std_0001.jpg --output crop_0001.jpg --debug
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class CropConfig:
    """Tunable parameters for optic disc detection and synthetic cropping."""

    # --- FOV mask (for standardized images) ---
    # Even after preprocessing, we need a mask to exclude any residual
    # black corners (UWF FOV is elliptical, bounding box is rectangular).
    mask_threshold: int = 15
    morph_kernel_size: int = 15

    # --- Optic disc detection ---
    # Gaussian blur kernel size (must be odd). Larger = smoother.
    od_blur_ksize: int = 51

    # Preferred color channel for OD detection.
    # 'red' saturates the disc in most UWF; 'green' gives best vessel contrast.
    od_channel: str = "red"

    # Expected optic disc diameter as a fraction of the FOV diameter.
    # Used for candidate scoring (~1.5mm disc / ~30mm FOV ≈ 0.05).
    od_diameter_fov_fraction: float = 0.05

    # Fraction of FOV radius to exclude from the OD search.
    # Masks out the outer ring where eyelid/scleral glow lives.
    peripheral_exclusion_ratio: float = 0.25

    # --- Macula estimation ---
    # Macula offset from OD in disc-diameters (temporal direction).
    macula_offset_dd: float = 2.5

    # Auto-detect laterality (OD vs OS) from disc position in the image.
    auto_laterality: bool = True

    # Fallback if auto-detect is off or ambiguous. 'OD' = right eye.
    default_laterality: str = "OD"

    # --- Synthetic crop ---
    # UWF field of view in degrees (Optos ~200°, Clarus ~133°).
    uwf_fov_degrees: float = 200.0

    # Target synthetic crop FOV in degrees.
    target_fov_degrees: float = 45.0

    # Output size of the final square crop in pixels.
    output_size: int = 512

    # JPEG quality for output (0-100).
    jpeg_quality: int = 95


# ──────────────────────────────────────────────
# FOV Mask (for standardized images)
# ──────────────────────────────────────────────

def build_fov_mask(image: np.ndarray, cfg: CropConfig) -> np.ndarray:
    """
    Build a binary mask of the retinal FOV.

    Even after tight bounding-box cropping in preprocessing, the UWF FOV
    is elliptical/irregular, so the rectangular image still has black
    corners. This mask identifies the actual retinal pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, cfg.mask_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.morph_kernel_size, cfg.morph_kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


# ──────────────────────────────────────────────
# Optic Disc Detection
# ──────────────────────────────────────────────

def _build_central_mask(mask: np.ndarray, exclusion_ratio: float) -> np.ndarray:
    """
    Erode the FOV mask to exclude the outer periphery.

    The outer ring of UWF images often contains bright eyelid/scleral
    reflections that are brighter than the optic disc. The OD is almost
    always in the central 50-75% of the FOV.
    """
    h, w = mask.shape[:2]
    fov_diameter = max(h, w)
    erode_px = int(fov_diameter * exclusion_ratio)

    if erode_px < 3:
        return mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    return cv2.erode(mask, kernel, iterations=1)


def _score_od_candidate(
    contour: np.ndarray,
    fov_diameter: float,
    expected_od_fraction: float,
) -> float:
    """
    Score a bright blob on how likely it is to be the optic disc.

    Considers circularity (OD is round) and size appropriateness
    (OD is ~3-7% of FOV diameter). Penalizes oversized blobs.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0 or area == 0:
        return 0.0

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    circularity = min(circularity, 1.0)

    blob_diameter = np.sqrt(4 * area / np.pi)
    blob_fraction = blob_diameter / (fov_diameter + 1e-6)

    # Gaussian penalty centered on expected OD fraction
    size_score = np.exp(
        -0.5 * ((blob_fraction - expected_od_fraction) / 0.03) ** 2
    )

    # Heavy penalty for blobs that are way too large (peripheral glow)
    if blob_fraction > 0.15:
        size_score *= 0.1

    return circularity * 0.5 + size_score * 0.5


def find_optic_disc(
    image: np.ndarray, mask: np.ndarray, cfg: CropConfig
) -> Tuple[int, int, float]:
    """
    Locate the optic disc center with peripheral exclusion and candidate scoring.

    Strategy:
        1. Exclude the outer periphery (eyelid/scleral glow zone).
        2. Search for bright, circular, appropriately-sized blobs.
        3. Try the preferred channel first, fall back to others.
        4. Progressively lower the brightness threshold if needed.

    Returns:
        (cx, cy):   Center coordinates of the optic disc.
        confidence: Detection confidence (0-1).
    """
    h, w = image.shape[:2]
    fov_diameter = max(h, w)

    central_mask = _build_central_mask(mask, cfg.peripheral_exclusion_ratio)

    channels = {
        "red": image[:, :, 2],
        "green": image[:, :, 1],
    }
    channel_order = [cfg.od_channel] + [
        ch for ch in channels if ch != cfg.od_channel
    ]

    best_cx, best_cy, best_score = w // 2, h // 2, 0.0

    for ch_name in channel_order:
        channel = channels.get(ch_name)
        if channel is None:
            continue

        for search_mask, search_label in [
            (central_mask, "central"),
            (mask, "full"),
        ]:
            masked_ch = cv2.bitwise_and(channel, channel, mask=search_mask)

            ksize = cfg.od_blur_ksize
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(masked_ch, (ksize, ksize), 0)

            max_val = blurred.max()
            if max_val == 0:
                continue

            for frac in [0.92, 0.85, 0.75]:
                thresh_val = int(max_val * frac)
                _, bright_mask = cv2.threshold(
                    blurred, thresh_val, 255, cv2.THRESH_BINARY
                )

                contours, _ = cv2.findContours(
                    bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue

                for cnt in contours:
                    score = _score_od_candidate(
                        cnt, fov_diameter, cfg.od_diameter_fov_fraction
                    )

                    if search_label == "central":
                        score *= 1.3
                    if ch_name == cfg.od_channel:
                        score *= 1.1

                    if score > best_score:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            best_cx = int(M["m10"] / M["m00"])
                            best_cy = int(M["m01"] / M["m00"])
                            best_score = score

                if best_score > 0.4:
                    break

            if search_label == "central" and best_score > 0.3:
                break

        if best_score > 0.5:
            break

    return best_cx, best_cy, min(best_score, 1.0)


# ──────────────────────────────────────────────
# Macula Estimation & Laterality
# ──────────────────────────────────────────────

def estimate_macula_center(
    image: np.ndarray,
    od_x: int,
    od_y: int,
    cfg: CropConfig,
) -> Tuple[int, int, str]:
    """
    Estimate the macula center from the optic disc position.

    The macula is temporal to the disc:
      - Right eye (OD): macula is LEFT of the disc.
      - Left eye (OS):  macula is RIGHT of the disc.

    Laterality is auto-detected from disc position (disc in right half →
    right eye) unless disabled in config.

    Returns:
        (mx, my):    Estimated macula center.
        laterality:  'OD' or 'OS'.
    """
    h, w = image.shape[:2]
    fov_diameter = max(w, h)

    dd_px = fov_diameter * cfg.od_diameter_fov_fraction
    offset_px = cfg.macula_offset_dd * dd_px

    if cfg.auto_laterality:
        laterality = "OD" if od_x > w / 2 else "OS"
    else:
        laterality = cfg.default_laterality

    if laterality == "OD":
        mx = int(od_x - offset_px)
    else:
        mx = int(od_x + offset_px)

    my = od_y  # Macula is roughly at the same vertical level

    mx = max(0, min(mx, w - 1))
    my = max(0, min(my, h - 1))

    return mx, my, laterality


# ──────────────────────────────────────────────
# Synthetic Crop
# ──────────────────────────────────────────────

def synthetic_crop(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    cfg: CropConfig,
) -> np.ndarray:
    """
    Extract a square crop simulating a `target_fov_degrees` field of view.

    Crop math:
        crop_fraction = target_fov / uwf_fov
        crop_radius_px = crop_fraction × (fov_diameter / 2)

    The crop is resized to `output_size × output_size`.
    Regions outside the image boundary are padded black.
    """
    h, w = image.shape[:2]
    fov_diameter = max(w, h)

    crop_fraction = cfg.target_fov_degrees / cfg.uwf_fov_degrees
    crop_radius = int((crop_fraction * fov_diameter) / 2)

    x1 = center_x - crop_radius
    y1 = center_y - crop_radius
    x2 = center_x + crop_radius
    y2 = center_y + crop_radius

    # Handle crops that extend beyond image boundaries
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    cropped = image[y1c:y2c, x1c:x2c]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    result = cv2.resize(
        cropped, (cfg.output_size, cfg.output_size),
        interpolation=cv2.INTER_AREA,
    )
    return result


# ──────────────────────────────────────────────
# Debug Visualization
# ──────────────────────────────────────────────

def save_debug_overlay(
    image: np.ndarray,
    od_x: int, od_y: int,
    mac_x: int, mac_y: int,
    cfg: CropConfig,
    output_path: str,
    laterality: str,
    confidence: float,
):
    """Save an annotated image showing OD, macula, and crop box."""
    vis = image.copy()
    h, w = vis.shape[:2]
    fov_diameter = max(w, h)

    # Optic disc (green circle)
    cv2.circle(vis, (od_x, od_y), 20, (0, 255, 0), 3)
    cv2.putText(
        vis, f"OD ({confidence:.2f})", (od_x + 25, od_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    # Macula estimate (magenta circle)
    cv2.circle(vis, (mac_x, mac_y), 15, (255, 0, 255), 3)
    cv2.putText(
        vis, "Macula (est)", (mac_x + 20, mac_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2,
    )

    # Crop box (yellow rectangle)
    crop_fraction = cfg.target_fov_degrees / cfg.uwf_fov_degrees
    crop_radius = int((crop_fraction * fov_diameter) / 2)
    cv2.rectangle(
        vis,
        (mac_x - crop_radius, mac_y - crop_radius),
        (mac_x + crop_radius, mac_y + crop_radius),
        (0, 255, 255), 3,
    )

    # Laterality label
    cv2.putText(
        vis, f"Eye: {laterality}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
    )

    cv2.imwrite(output_path, vis)


# ──────────────────────────────────────────────
# Processing Results
# ──────────────────────────────────────────────

@dataclass
class CropResult:
    filename: str
    input_size: Tuple[int, int]          # (width, height) of standardized input
    optic_disc: Tuple[int, int]          # (x, y) in standardized image
    od_confidence: float
    macula_estimate: Tuple[int, int]     # (x, y) in standardized image
    laterality: str                      # 'OD' or 'OS'
    crop_center: Tuple[int, int]         # actual center used for the crop
    crop_radius_px: int                  # half-side of the crop square
    output_size: Tuple[int, int]         # final output dimensions
    success: bool
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Single Image Processing
# ──────────────────────────────────────────────

def crop_single(
    input_path: str,
    output_path: str,
    cfg: CropConfig,
    debug_dir: Optional[str] = None,
    center_on: str = "macula",
) -> CropResult:
    """
    Full synthetic crop pipeline for one standardized image.

    Args:
        center_on: What to center the crop on.
            'macula'     – estimated macula (recommended, standard fundus equivalent)
            'optic_disc' – directly on the detected OD
            'fov_center' – geometric center (no anatomical anchoring)
    """
    filename = os.path.basename(input_path)

    try:
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        h, w = image.shape[:2]
        fov_diameter = max(w, h)

        # Build FOV mask
        mask = build_fov_mask(image, cfg)

        # Detect optic disc
        od_x, od_y, od_conf = find_optic_disc(image, mask, cfg)

        # Estimate macula
        mac_x, mac_y, laterality = estimate_macula_center(
            image, od_x, od_y, cfg
        )

        # Choose crop center
        if center_on == "macula":
            cx, cy = mac_x, mac_y
        elif center_on == "optic_disc":
            cx, cy = od_x, od_y
        elif center_on == "fov_center":
            cx, cy = w // 2, h // 2
        else:
            raise ValueError(f"Unknown center_on mode: {center_on}")

        # Crop
        crop_fraction = cfg.target_fov_degrees / cfg.uwf_fov_degrees
        crop_radius = int((crop_fraction * fov_diameter) / 2)

        result_img = synthetic_crop(image, cx, cy, cfg)

        # Save
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(
                output_path, result_img,
                [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
            )
        elif ext == ".png":
            cv2.imwrite(
                output_path, result_img,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
        else:
            cv2.imwrite(output_path, result_img)

        # Debug overlay
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"debug_{filename}")
            save_debug_overlay(
                image, od_x, od_y, mac_x, mac_y,
                cfg, debug_path, laterality, od_conf,
            )

        return CropResult(
            filename=filename,
            input_size=(w, h),
            optic_disc=(od_x, od_y),
            od_confidence=round(od_conf, 3),
            macula_estimate=(mac_x, mac_y),
            laterality=laterality,
            crop_center=(cx, cy),
            crop_radius_px=crop_radius,
            output_size=(cfg.output_size, cfg.output_size),
            success=True,
        )

    except Exception as e:
        return CropResult(
            filename=filename,
            input_size=(0, 0),
            optic_disc=(0, 0),
            od_confidence=0.0,
            macula_estimate=(0, 0),
            laterality="unknown",
            crop_center=(0, 0),
            crop_radius_px=0,
            output_size=(0, 0),
            success=False,
            error=str(e),
        )


# ──────────────────────────────────────────────
# Batch Processing
# ──────────────────────────────────────────────

def batch_crop(
    input_dir: str,
    output_dir: str,
    cfg: CropConfig,
    debug: bool = False,
    center_on: str = "macula",
) -> list:
    """Crop all standardized images in a directory (recursively)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print(f"Scanning for images in {input_dir}...")
    files = sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        print(f"No images found in {input_dir}")
        return []

    results = []
    low_conf = []

    for i, fpath in enumerate(files, 1):
        rel_path = fpath.relative_to(input_path)
        dest_path = output_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve structure in debug folder if enabled
        current_debug_dir = None
        if debug:
            current_debug_dir = str(output_path / "_debug" / rel_path.parent)

        print(f"[{i}/{len(files)}] {rel_path}...", end=" ", flush=True)

        result = crop_single(str(fpath), str(dest_path), cfg, current_debug_dir, center_on)

        if result.success:
            conf_marker = ""
            if result.od_confidence < 0.5:
                conf_marker = " ⚠ LOW CONFIDENCE"
                low_conf.append(result.filename)
            print(
                f"OK  OD=({result.optic_disc[0]},{result.optic_disc[1]}) "
                f"conf={result.od_confidence:.2f} "
                f"eye={result.laterality}{conf_marker}"
            )
        else:
            print(f"FAIL — {result.error}")

        results.append(result)

    # Save metadata
    meta_path = os.path.join(output_dir, "crop_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": [asdict(r) for r in results],
            },
            f, indent=2,
        )
    print(f"\nMetadata saved to {meta_path}")

    successes = sum(1 for r in results if r.success)
    print(f"Done: {successes}/{len(results)} images cropped successfully.")

    if low_conf:
        print(f"\n⚠  {len(low_conf)} images had low OD confidence (<0.5):")
        for fn in low_conf:
            print(f"   - {fn}")
        print("   Review these in the _debug folder.")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Synthetic 45° crop from standardized UWF fundus images"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str,
                       help="Single standardized image path")
    group.add_argument("--input_dir", type=str,
                       help="Directory of standardized images (from 1_preprocess.py)")

    parser.add_argument("--output", type=str,
                        help="Output path (single image mode)")
    parser.add_argument("--output_dir", type=str, default="./cropped_45deg",
                        help="Output directory (batch mode, default: ./cropped_45deg)")

    parser.add_argument("--debug", action="store_true",
                        help="Save debug overlays showing OD, macula, and crop box")

    parser.add_argument("--center_on", type=str, default="macula",
                        choices=["macula", "optic_disc", "fov_center"],
                        help="Anatomical anchor for the crop center (default: macula)")

    # Config overrides
    parser.add_argument("--uwf_fov", type=float, default=200.0,
                        help="UWF FOV in degrees (Optos=200, Clarus=133)")
    parser.add_argument("--target_fov", type=float, default=45.0,
                        help="Target crop FOV in degrees (default: 45)")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Output crop size in pixels, square (default: 512)")
    parser.add_argument("--od_channel", type=str, default="red",
                        choices=["red", "green"],
                        help="Color channel for OD detection (default: red)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG output quality, 0-100 (default: 95)")

    args = parser.parse_args()

    cfg = CropConfig(
        uwf_fov_degrees=args.uwf_fov,
        target_fov_degrees=args.target_fov,
        output_size=args.output_size,
        od_channel=args.od_channel,
        jpeg_quality=args.jpeg_quality,
    )

    if args.input:
        out = args.output or f"crop_{os.path.basename(args.input)}"
        debug_dir = "./_debug" if args.debug else None
        result = crop_single(args.input, out, cfg, debug_dir, args.center_on)
        if result.success:
            print(f"Saved: {out}")
            print(f"  OD detected at {result.optic_disc} (conf={result.od_confidence})")
            print(f"  Laterality: {result.laterality}")
            print(f"  Crop center: {result.crop_center}")
            print(f"  Crop radius: {result.crop_radius_px}px")
        else:
            print(f"Failed: {result.error}", file=sys.stderr)
            sys.exit(1)
    else:
        batch_crop(
            args.input_dir, args.output_dir, cfg, args.debug, args.center_on
        )


if __name__ == "__main__":
    main()