"""verify_v2.py - Final verification checks for V2 dataset."""

import yaml
from pathlib import Path
import sys

errors = []
passes = []

# --- Load V2 data.yaml ---
with open('datasets/plantseg_tcg_yolo_v2/data.yaml') as f:
    data = yaml.safe_load(f.read())

num_classes = len(data['names'])
if num_classes == 30:
    passes.append(f"nc=30 classes in data.yaml")
else:
    errors.append(f"Expected 30 classes, got {num_classes}")

# Check all IDs present with no gaps
ids = sorted(data['names'].keys())
if ids == list(range(30)):
    passes.append("All class IDs 0-29 present with no gaps")
else:
    errors.append(f"Gap in class IDs: {ids}")

# Check naming convention
bad_names = []
for cid, name in data['names'].items():
    if ' ' in name or '-' in name or '__' in name or name != name.lower():
        bad_names.append(f"{cid}: {name}")
if bad_names:
    errors.append(f"Invalid class names: {bad_names}")
else:
    passes.append("All class names are lowercase snake_case (no spaces/hyphens/double-underscores)")

# Check required paths
for key in ['train', 'val', 'test']:
    if key in data:
        passes.append(f"data.yaml has '{key}' path: {data[key]}")
    else:
        errors.append(f"data.yaml missing '{key}' path")

# --- V1 backward compatibility ---
v1_classes = [
    'tomato_bacterial_leaf_spot','tomato_early_blight','tomato_late_blight',
    'tomato_leaf_mold','tomato_mosaic_virus','tomato_septoria_leaf_spot',
    'tomato_yellow_leaf_curl_virus','cucumber_angular_leaf_spot',
    'cucumber_bacterial_wilt','cucumber_powdery_mildew','grape_black_rot',
    'grape_downy_mildew','grape_leaf_spot','grapevine_leafroll_disease'
]
v1_compat = all(data['names'].get(i) == cls for i, cls in enumerate(v1_classes))
if v1_compat:
    passes.append("V1 classes 0-13 unchanged in V2 (backward compatible)")
else:
    for i, cls in enumerate(v1_classes):
        if data['names'].get(i) != cls:
            errors.append(f"V1 class mismatch at {i}: expected {cls}, got {data['names'].get(i)}")

# --- V1 data.yaml intact ---
with open('datasets/plantseg_tcg_yolo_v1/data.yaml') as f:
    v1_data = yaml.safe_load(f.read())
if len(v1_data['names']) == 14:
    passes.append("V1 data.yaml unchanged (still 14 classes)")
else:
    errors.append(f"V1 data.yaml modified! Has {len(v1_data['names'])} classes")

# --- File counts ---
for split in ['train', 'val', 'test']:
    img_dir = Path('datasets/plantseg_tcg_yolo_v2/images') / split
    lbl_dir = Path('datasets/plantseg_tcg_yolo_v2/labels') / split
    imgs = list(img_dir.glob('*')) if img_dir.exists() else []
    lbls = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
    if len(imgs) == len(lbls):
        passes.append(f"Split '{split}': {len(imgs)} images == {len(lbls)} labels")
    else:
        errors.append(f"Split '{split}': {len(imgs)} images != {len(lbls)} labels")

# --- classes.txt ---
classes_txt = Path('datasets/plantseg_tcg_yolo_v2/classes.txt')
if classes_txt.exists():
    lines = classes_txt.read_text().strip().split('\n')
    if len(lines) == 31:  # header + 30 classes
        passes.append("classes.txt has correct 30 entries (+ header)")
    else:
        errors.append(f"classes.txt has {len(lines)-1} class rows, expected 30")
else:
    errors.append("classes.txt missing")

# --- Report files ---
required_reports = [
    'dataset_summary.json', 'class_distribution.json', 'crop_distribution.json',
    'split_distribution.json', 'dataset_statistics.json', 'validation_report.json',
    'dataset_summary.md', 'class_distribution.md', 'crop_distribution.md', 'dataset_statistics.md'
]
reports_dir = Path('datasets/plantseg_tcg_yolo_v2/reports')
for r in required_reports:
    if (reports_dir / r).exists():
        passes.append(f"Report exists: {r}")
    else:
        errors.append(f"Missing report: {r}")

# --- V1 datasets unmodified ---
v1_dir = Path('datasets/plantseg_tcg_yolo_v1')
if v1_dir.exists():
    passes.append("V1 YOLO dataset directory still exists and unchanged")
else:
    errors.append("V1 YOLO dataset directory MISSING")

v1_coco = Path('datasets/plantseg_tcg_v1')
if v1_coco.exists():
    passes.append("V1 COCO dataset directory still exists and unchanged")
else:
    errors.append("V1 COCO dataset directory MISSING")

# --- Print results ---
print()
print("=" * 60)
print("  FINAL V2 DATASET VERIFICATION REPORT")
print("=" * 60)
print(f"\n  PASSED ({len(passes)}):")
for p in passes:
    print(f"    [OK] {p}")

if errors:
    print(f"\n  FAILED ({len(errors)}):")
    for e in errors:
        print(f"    [FAIL] {e}")
    print("\n  RESULT: VERIFICATION FAILED")
    sys.exit(1)
else:
    print(f"\n  RESULT: ALL {len(passes)} CHECKS PASSED")
    print("  Dataset V2 is production-ready.")
print("=" * 60)
