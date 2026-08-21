#!/usr/bin/env python3
"""
visualize_labels.py
-------------------
Samples images from each split, overlays the segmentation polygons and labels,
and exports them to reports/visual_validation/ for manual inspection.

Samples:
  - 10 training images
  - 5 validation images
  - 5 test images

Usage:
    python scripts/visualize_labels.py \
        --input datasets/plantseg_tcg_yolo_v1 \
        --output datasets/plantseg_tcg_yolo_v1/reports/visual_validation \
        --verbose
"""

import argparse
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("ERROR: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# Class list to resolve class IDs to names
DISEASE_CLASSES = [
    "tomato_bacterial_leaf_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
    "cucumber_angular_leaf_spot",
    "cucumber_bacterial_wilt",
    "cucumber_powdery_mildew",
    "grape_black_rot",
    "grape_downy_mildew",
    "grape_leaf_spot",
    "grapevine_leafroll_disease",
]

# Color map for 14 classes (curated Sleek/Harmonious colors)
COLORS = [
    (239, 71, 111),   # Red-pink
    (247, 140, 107),  # Coral
    (255, 209, 102),  # Yellow
    (6, 214, 160),    # Teal
    (17, 138, 178),   # Blue
    (7, 59, 76),      # Dark Blue
    (131, 56, 236),   # Purple
    (251, 86, 196),   # Pink
    (58, 125, 68),    # Forest Green
    (112, 224, 0),    # Lime Green
    (224, 122, 95),   # Sienna
    (61, 90, 128),    # Slate
    (152, 193, 217),  # Light slate
    (238, 108, 77),   # Terracotta
]


def overlay_polygons(img_path: Path, label_path: Path, out_path: Path) -> bool:
    """Read image and label file, draw polygons, save to out_path."""
    try:
        with Image.open(img_path) as img:
            # Create transparent overlay layer
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            w, h = img.size

            if not label_path.exists():
                return False

            with open(label_path, encoding="utf-8") as f:
                lines = f.read().strip().split("\n")

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # Map back to absolute pixels
                poly_pts = []
                for idx in range(0, len(coords), 2):
                    px = coords[idx] * w
                    py = coords[idx+1] * h
                    poly_pts.append((px, py))

                if class_id < len(COLORS):
                    color = COLORS[class_id]
                else:
                    color = (255, 255, 255)

                # Draw polygon fill on overlay layer (semi-transparent)
                draw.polygon(poly_pts, fill=color + (80,), outline=color + (255,), width=2)

                # Draw class label on overlay near the first point of the polygon
                class_name = DISEASE_CLASSES[class_id] if class_id < len(DISEASE_CLASSES) else f"Class_{class_id}"
                draw.text(poly_pts[0], class_name, fill=(255, 255, 255, 255), stroke_fill=(0, 0, 0, 255), stroke_width=1)

            # Composite the image with the overlay layer
            final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
            # Save as RGB format (typically JPG)
            final_img.convert("RGB").save(out_path, "JPEG")
            return True
    except Exception as e:
        print(f"    ✗ Error overlaying {img_path.name}: {e}")
        return False


def run(args):
    dataset_root = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("\n══ YOLO Dataset Visualizer ══")

    # Set seed for reproducibility
    random.seed(42)

    sample_requirements = {
        "train": 10,
        "val": 5,
        "test": 5,
    }

    generated_samples = 0

    for split, count in sample_requirements.items():
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists():
            continue

        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not images:
            continue

        # Filter out images without valid annotations if we want to ensure visual overlay
        valid_pairs = []
        for img in images:
            txt_path = lbl_dir / (img.stem + ".txt")
            if txt_path.exists() and txt_path.stat().st_size > 0:
                valid_pairs.append((img, txt_path))

        if len(valid_pairs) < count:
            # Fallback to all images if not enough annotated ones
            valid_pairs = [(img, lbl_dir / (img.stem + ".txt")) for img in images]

        # Draw random samples
        sample = random.sample(valid_pairs, min(count, len(valid_pairs)))

        if args.verbose:
            print(f"  Split [{split}]: Sampling {len(sample)} images for visualization …")

        for img, lbl in sample:
            out_img_path = out_dir / f"{split}_{img.stem}_overlay.jpg"
            success = overlay_polygons(img, lbl, out_img_path)
            if success:
                generated_samples += 1
                if args.verbose:
                    print(f"    ✓ Exported overlay: {out_img_path.name}")

    if args.verbose or True:
        print(f"  ✓ Exported {generated_samples} visual validation samples to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO Segmentation labels")
    parser.add_argument("--input",   required=True, help="Path to YOLO dataset root")
    parser.add_argument("--output",  required=True, help="Path to reports/visual_validation destination folder")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
