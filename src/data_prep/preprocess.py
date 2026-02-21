"""
Step 1: UWF Fundus Image Preprocessing & Standardization
=========================================================
Removes black borders / padding from UWF fundus images so the retinal
field of view (FOV) fills the canvas edge-to-edge. Handles both zoomed-in
images (no borders) and zoomed-out images (heavy black padding) identically.

Output: A directory of standardized images where every image has the FOV
        tightly cropped, plus a metadata JSON recording what was done.

Requirements:
    pip install opencv-python-headless numpy

Usage:
    # Batch process (most common)
    python 1_preprocess.py --input_dir ./raw_images --output_dir ./standardized

    # Single image
    python 1_preprocess.py --input image_0001.jpg --output std_0001.jpg

    # Adjust threshold if faint sensor glow is included or real retina is clipped
    python 1_preprocess.py --input_dir ./raw --output_dir ./std --threshold 20

    # Resize all standardized outputs to a uniform dimension
    python 1_preprocess.py --input_dir ./raw --output_dir ./std --resize 2048
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
class PreprocessConfig:
    """Tunable parameters for border removal and standardization."""

    # Threshold for separating retina from black background (0-255).
    # Pixels darker than this are treated as background.
    # Increase if faint sensor glow at edges is being included.
    # Decrease if dark retinal regions are being clipped.
    border_threshold: int = 15

    # Morphological kernel size for cleaning the binary mask.
    # Larger = more aggressive noise removal (fills small holes,
    # removes specks). Must be odd or will be rounded up.
    morph_kernel_size: int = 15

    # Minimum contour area as a fraction of total image area.
    # Contours smaller than this are treated as dust/artifacts.
    min_fov_area_ratio: float = 0.05

    # If set, resize the longest edge of the output to this value
    # (preserving aspect ratio). None = keep original resolution.
    resize_longest_edge: Optional[int] = None

    # JPEG quality for output images (0-100). Only applies to .jpg/.jpeg.
    jpeg_quality: int = 95


# ──────────────────────────────────────────────
# Core: FOV Detection & Border Removal
# ──────────────────────────────────────────────

def detect_fov_mask(
    image: np.ndarray, cfg: PreprocessConfig
) -> Tuple[np.ndarray, list]:
    """
    Create a binary mask separating the retinal FOV from black background.

    Returns:
        mask:     Binary mask (255 = retina, 0 = background), same size as input.
        contours: List of valid contours found (largest first).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, cfg.border_threshold, 255, cv2.THRESH_BINARY
    )

    # Morphological cleanup: close small gaps, then remove small specks
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.morph_kernel_size, cfg.morph_kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours and filter by minimum area
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = cfg.min_fov_area_ratio * image.shape[0] * image.shape[1]
    valid = sorted(
        [c for c in contours if cv2.contourArea(c) >= min_area],
        key=cv2.contourArea,
        reverse=True,
    )

    return binary, valid


def remove_black_borders(
    image: np.ndarray, cfg: PreprocessConfig
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Detect the retinal FOV and crop away black padding.

    Returns:
        cropped_image: The tightly cropped retinal image.
        cropped_mask:  Binary FOV mask (same size as cropped_image).
        bbox:          (x, y, w, h) of the FOV bounding box in the original image.
    """
    binary, valid_contours = detect_fov_mask(image, cfg)

    if not valid_contours:
        raise ValueError(
            "No FOV detected. The image may be completely black, or "
            "`border_threshold` may be too high (currently "
            f"{cfg.border_threshold}). Try lowering it."
        )

    largest = valid_contours[0]
    x, y, w, h = cv2.boundingRect(largest)

    cropped_image = image[y : y + h, x : x + w].copy()
    cropped_mask = binary[y : y + h, x : x + w]

    return cropped_image, cropped_mask, (x, y, w, h)


def resize_image(
    image: np.ndarray, longest_edge: int
) -> Tuple[np.ndarray, float]:
    """
    Resize so the longest edge equals `longest_edge`, preserving aspect ratio.

    Returns:
        resized: The resized image.
        scale:   The scale factor applied (new / old).
    """
    h, w = image.shape[:2]
    current_longest = max(h, w)

    if current_longest <= longest_edge:
        return image, 1.0

    scale = longest_edge / current_longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ──────────────────────────────────────────────
# Processing Results
# ──────────────────────────────────────────────

@dataclass
class PreprocessResult:
    filename: str
    original_size: Tuple[int, int]       # (width, height)
    fov_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original
    standardized_size: Tuple[int, int]   # (width, height) after crop + optional resize
    border_pixels_removed: Tuple[int, int, int, int]  # (top, bottom, left, right)
    scale_factor: float                  # 1.0 if no resize applied
    fov_area_fraction: float             # fraction of original canvas that was FOV
    success: bool
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Single Image Processing
# ──────────────────────────────────────────────

def preprocess_single(
    input_path: str,
    output_path: str,
    cfg: PreprocessConfig,
) -> PreprocessResult:
    """
    Preprocess one image: remove borders, optionally resize, save.
    """
    filename = os.path.basename(input_path)

    try:
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        orig_h, orig_w = image.shape[:2]

        # Remove black borders
        cropped, mask, (bx, by, bw, bh) = remove_black_borders(image, cfg)

        # Calculate how much border was removed on each side
        border_top = by
        border_left = bx
        border_bottom = orig_h - (by + bh)
        border_right = orig_w - (bx + bw)

        # FOV area as fraction of original canvas
        fov_pixels = cv2.countNonZero(mask)
        fov_fraction = fov_pixels / (orig_h * orig_w)

        # Optional resize
        scale = 1.0
        if cfg.resize_longest_edge is not None:
            cropped, scale = resize_image(cropped, cfg.resize_longest_edge)

        std_h, std_w = cropped.shape[:2]

        # Save with appropriate quality
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(
                output_path, cropped,
                [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
            )
        elif ext == ".png":
            cv2.imwrite(
                output_path, cropped,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
        else:
            cv2.imwrite(output_path, cropped)

        return PreprocessResult(
            filename=filename,
            original_size=(orig_w, orig_h),
            fov_bbox=(bx, by, bw, bh),
            standardized_size=(std_w, std_h),
            border_pixels_removed=(border_top, border_bottom, border_left, border_right),
            scale_factor=round(scale, 4),
            fov_area_fraction=round(fov_fraction, 4),
            success=True,
        )

    except Exception as e:
        return PreprocessResult(
            filename=filename,
            original_size=(0, 0),
            fov_bbox=(0, 0, 0, 0),
            standardized_size=(0, 0),
            border_pixels_removed=(0, 0, 0, 0),
            scale_factor=0.0,
            fov_area_fraction=0.0,
            success=False,
            error=str(e),
        )


# ──────────────────────────────────────────────
# Batch Processing
# ──────────────────────────────────────────────

def batch_preprocess(
    input_dir: str,
    output_dir: str,
    cfg: PreprocessConfig,
) -> list:
    """Preprocess all images in a directory (recursively)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    files = sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        print(f"No images found in {input_dir}")
        return []

    results = []
    for i, fpath in enumerate(files, 1):
        rel_path = fpath.relative_to(input_path)
        dest_path = output_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(files)}] {rel_path}...", end=" ")

        result = preprocess_single(str(fpath), str(dest_path), cfg)

        if result.success:
            ow, oh = result.original_size
            sw, sh = result.standardized_size
            t, b, l, r = result.border_pixels_removed
            total_border = t + b + l + r
            if total_border > 0:
                print(
                    f"OK  {ow}x{oh} → {sw}x{sh}  "
                    f"(removed {t}px top, {b}px bot, {l}px left, {r}px right)"
                )
            else:
                print(f"OK  {ow}x{oh} → {sw}x{sh}  (no borders detected)")
        else:
            print(f"FAIL — {result.error}")

        results.append(result)

    # Save metadata
    meta_path = os.path.join(output_dir, "preprocess_metadata.json")
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
    print(f"Done: {successes}/{len(results)} images standardized successfully.")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Remove black borders and standardize UWF fundus images"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Single image path")
    group.add_argument("--input_dir", type=str, help="Directory of raw images")

    parser.add_argument("--output", type=str,
                        help="Output path (single image mode)")
    parser.add_argument("--output_dir", type=str, default="./standardized",
                        help="Output directory (batch mode, default: ./standardized)")

    parser.add_argument("--threshold", type=int, default=15,
                        help="Black border detection threshold, 0-255 (default: 15)")
    parser.add_argument("--morph_kernel", type=int, default=15,
                        help="Morphological cleanup kernel size (default: 15)")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize longest edge to this value (default: no resize)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG output quality, 0-100 (default: 95)")

    args = parser.parse_args()

    cfg = PreprocessConfig(
        border_threshold=args.threshold,
        morph_kernel_size=args.morph_kernel,
        resize_longest_edge=args.resize,
        jpeg_quality=args.jpeg_quality,
    )

    if args.input:
        out = args.output or f"std_{os.path.basename(args.input)}"
        result = preprocess_single(args.input, out, cfg)
        if result.success:
            ow, oh = result.original_size
            sw, sh = result.standardized_size
            print(f"Saved: {out}")
            print(f"  Original:     {ow} x {oh}")
            print(f"  Standardized: {sw} x {sh}")
            print(f"  FOV coverage: {result.fov_area_fraction:.1%} of original canvas")
            print(f"  Borders removed (T/B/L/R): {result.border_pixels_removed}")
        else:
            print(f"Failed: {result.error}", file=sys.stderr)
            sys.exit(1)
    else:
        batch_preprocess(args.input_dir, args.output_dir, cfg)


if __name__ == "__main__":
    main()