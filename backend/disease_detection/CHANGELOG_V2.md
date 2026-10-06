# CHANGELOG V2 — PlantSeg Dataset Expansion

## Summary

Dataset expanded from **V1** (Tomato, Cucumber, Grape — 14 classes) to **V2** (+ Banana, Corn, Soybean — **30 classes**).

---

## New Crops Added

| Crop | Diseases Added | Class IDs |
|---|---|---|
| **Banana** | anthracnose, black leaf streak, bunchy top, cigar end rot, cordana leaf spot, panama disease | 14–19 |
| **Corn** | northern leaf blight, gray leaf spot, rust, smut | 20–23 |
| **Soybean** | bacterial blight, brown spot, downy mildew, frog eye leaf spot, mosaic, rust | 24–29 |

---

## New Diseases (16 total)

### Banana (6)
- `banana_anthracnose` → class ID **14**
- `banana_black_leaf_streak` → class ID **15**
- `banana_bunchy_top` → class ID **16**
- `banana_cigar_end_rot` → class ID **17**
- `banana_cordana_leaf_spot` → class ID **18**
- `banana_panama_disease` → class ID **19**

### Corn (4)
- `corn_northern_leaf_blight` → class ID **20**
- `corn_gray_leaf_spot` → class ID **21**
- `corn_rust` → class ID **22**
- `corn_smut` → class ID **23**

### Soybean (6)
- `soybean_bacterial_blight` → class ID **24**
- `soybean_brown_spot` → class ID **25**
- `soybean_downy_mildew` → class ID **26**
- `soybean_frog_eye_leaf_spot` → class ID **27**
- `soybean_mosaic` → class ID **28**
- `soybean_rust` → class ID **29**

---

## Complete Class ID Mapping (V2)

| ID | Class Name | Crop | Status |
|---|---|---|---|
| 0 | `tomato_bacterial_leaf_spot` | Tomato | ✅ V1 unchanged |
| 1 | `tomato_early_blight` | Tomato | ✅ V1 unchanged |
| 2 | `tomato_late_blight` | Tomato | ✅ V1 unchanged |
| 3 | `tomato_leaf_mold` | Tomato | ✅ V1 unchanged |
| 4 | `tomato_mosaic_virus` | Tomato | ✅ V1 unchanged |
| 5 | `tomato_septoria_leaf_spot` | Tomato | ✅ V1 unchanged |
| 6 | `tomato_yellow_leaf_curl_virus` | Tomato | ✅ V1 unchanged |
| 7 | `cucumber_angular_leaf_spot` | Cucumber | ✅ V1 unchanged |
| 8 | `cucumber_bacterial_wilt` | Cucumber | ✅ V1 unchanged |
| 9 | `cucumber_powdery_mildew` | Cucumber | ✅ V1 unchanged |
| 10 | `grape_black_rot` | Grape | ✅ V1 unchanged |
| 11 | `grape_downy_mildew` | Grape | ✅ V1 unchanged |
| 12 | `grape_leaf_spot` | Grape | ✅ V1 unchanged |
| 13 | `grapevine_leafroll_disease` | Grape | ✅ V1 unchanged |
| 14 | `banana_anthracnose` | Banana | 🆕 New in V2 |
| 15 | `banana_black_leaf_streak` | Banana | 🆕 New in V2 |
| 16 | `banana_bunchy_top` | Banana | 🆕 New in V2 |
| 17 | `banana_cigar_end_rot` | Banana | 🆕 New in V2 |
| 18 | `banana_cordana_leaf_spot` | Banana | 🆕 New in V2 |
| 19 | `banana_panama_disease` | Banana | 🆕 New in V2 |
| 20 | `corn_northern_leaf_blight` | Corn | 🆕 New in V2 |
| 21 | `corn_gray_leaf_spot` | Corn | 🆕 New in V2 |
| 22 | `corn_rust` | Corn | 🆕 New in V2 |
| 23 | `corn_smut` | Corn | 🆕 New in V2 |
| 24 | `soybean_bacterial_blight` | Soybean | 🆕 New in V2 |
| 25 | `soybean_brown_spot` | Soybean | 🆕 New in V2 |
| 26 | `soybean_downy_mildew` | Soybean | 🆕 New in V2 |
| 27 | `soybean_frog_eye_leaf_spot` | Soybean | 🆕 New in V2 |
| 28 | `soybean_mosaic` | Soybean | 🆕 New in V2 |
| 29 | `soybean_rust` | Soybean | 🆕 New in V2 |

---

## Dataset Statistics

### Total Images
| Version | Images | Classes | Crops |
|---|---|---|---|
| V1 | 1,416 | 14 | 3 |
| **V2** | **2,665** | **30** | **6** |
| Delta | +1,249 | +16 | +3 |

### V2 Split Distribution
| Split | Images | Labels | Polygons |
|---|---|---|---|
| train | 1,864 | 1,864 | ~12,900 |
| val | 275 | 275 | ~1,900 |
| test | 526 | 526 | ~3,160 |
| **Total** | **2,665** | **2,665** | **17,961** |

### V2 Class Polygon Counts
| Class | Total Polygons |
|---|---|
| tomato_bacterial_leaf_spot | 1,327 |
| tomato_early_blight | 1,730 |
| tomato_late_blight | 321 |
| tomato_leaf_mold | 629 |
| tomato_mosaic_virus | 247 |
| tomato_septoria_leaf_spot | 1,320 |
| tomato_yellow_leaf_curl_virus | 601 |
| cucumber_angular_leaf_spot | 1,503 |
| cucumber_bacterial_wilt | 497 |
| cucumber_powdery_mildew | 920 |
| grape_black_rot | 914 |
| grape_downy_mildew | 781 |
| grape_leaf_spot | 380 |
| grapevine_leafroll_disease | 65 |
| banana_anthracnose | 495 |
| banana_black_leaf_streak | 452 |
| banana_bunchy_top | 246 |
| banana_cigar_end_rot | 343 |
| banana_cordana_leaf_spot | 150 |
| banana_panama_disease | 237 |
| corn_northern_leaf_blight | 824 |
| corn_gray_leaf_spot | 833 |
| corn_rust | 437 |
| corn_smut | 203 |
| soybean_bacterial_blight | 174 |
| soybean_brown_spot | 154 |
| soybean_downy_mildew | 216 |
| soybean_frog_eye_leaf_spot | 1,474 |
| soybean_mosaic | 291 |
| soybean_rust | 197 |

---

## Folder Changes

### New Folders Created
```
datasets/
  plantseg_v2/               ← COCO format V2 (images + masks + metadata)
    images/
      train/
      val/
      test/
    annotations/
      train/
      val/
      test/
    coco/
      annotation_train.json
      annotation_val.json
      annotation_test.json
    metadata/
      subset_metadata.csv

  plantseg_tcg_yolo_v2/      ← YOLO11 Segmentation format V2
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    annotations/
      train.json
      val.json
      test.json
    data.yaml
    classes.txt
    reports/
      dataset_summary.json
      class_distribution.json
      crop_distribution.json
      split_distribution.json
      dataset_statistics.json
      validation_report.json
      dataset_summary.md
      class_distribution.md
      crop_distribution.md
      dataset_statistics.md
```

### Unchanged Folders (V1 preserved as-is)
```
datasets/
  plantseg_raw/              ← NEVER modified
  plantseg_tcg_v1/           ← V1 COCO format — UNCHANGED
  plantseg_tcg_yolo_v1/      ← V1 YOLO format — UNCHANGED
```

---

## New Scripts Added

| Script | Purpose |
|---|---|
| `scripts/build_v2_dataset.py` | Full V2 dataset pipeline (filter, deduplicate, copy, convert to YOLO) |
| `scripts/validate_v2_dataset.py` | 12-point validation for V2 YOLO dataset |
| `scripts/generate_v2_reports.py` | All JSON + Markdown report generation for V2 |

---

## Configuration Changes

### `datasets/plantseg_tcg_yolo_v2/data.yaml` (NEW)
- `nc: 30` classes
- Paths: `images/train`, `images/val`, `images/test`
- All 30 names in snake_case, gapless IDs 0–29

### No Changes to V1 Files
- `datasets/plantseg_tcg_yolo_v1/data.yaml` — **UNCHANGED**
- `app/main.py` — **UNCHANGED**
- `app/services/yolo_detector.py` — **UNCHANGED**
- `scripts/convert_to_yolo.py` — **UNCHANGED**
- All existing training scripts — **UNCHANGED**

---

## Compatibility Notes

### Backward Compatibility
- V1 class IDs **0–13 are identical** in V2. Any model trained on V1 will still produce correct predictions for classes 0–13.
- The FastAPI service (`app/main.py`, `app/services/yolo_detector.py`) dynamically reads class names from the loaded model, so it requires no modification.
- To train V2, point `--data` to `datasets/plantseg_tcg_yolo_v2/data.yaml`.

### Training V2 Model
```bash
# Train a new V2 model (does NOT affect the existing V1 trained model)
yolo segment train \
  data=datasets/plantseg_tcg_yolo_v2/data.yaml \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=640 \
  project=models/trained_v2 \
  name=plantseg_v2
```

### Deduplication Notes
- **58 duplicate images** were removed during V2 build (SHA256 exact match).
- Deduplication priority: `train > val > test` (same policy as V1).
- No perceptual (pHash) deduplication performed at build time; cross-split SHA256 overlap was confirmed to be 0.

### Validation Summary
- **0 failures**, **29 warnings**
- All 29 warnings are **zero-area / degenerate polygons** inherited from the original PlantSeg raw dataset. These polygons have all coordinates identical or fewer than 3 unique points. This is a known upstream data quality issue. YOLO training skips degenerate polygons automatically.

---

## Date
**2026-07-18**

## Author
Automated by `build_v2_dataset.py` pipeline.
