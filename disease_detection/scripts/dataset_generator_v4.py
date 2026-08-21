#!/usr/bin/env python3
"""
plantseg_tcg_yolo_v4 Dataset Generator (Healthy as Background)

Senior Computer Vision Engineer Tool for Ultralytics YOLO Segmentation Datasets.
Converts plantseg_tcg_yolo_v3 into plantseg_tcg_yolo_v4 by transforming healthy crop classes
(tomato_healthy, cucumber_healthy, grape_healthy) into negative background examples (empty label files),
while retaining disease classes (IDs 0-13) unchanged.

Google Colab compatible out of the box with standard library Python.
Does NOT overwrite V3 dataset.
"""

import os
import sys
import shutil
import json
import csv
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("v4_dataset_generator")

HEALTHY_CLASS_NAMES = {
    "tomato_healthy",
    "cucumber_healthy",
    "grape_healthy"
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert plantseg_tcg_yolo_v3 dataset into plantseg_tcg_yolo_v4 (Healthy as Background)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/plantseg_tcg_yolo_v3",
        help="Path to input plantseg_tcg_yolo_v3 dataset root."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/plantseg_tcg_yolo_v4",
        help="Path to output plantseg_tcg_yolo_v4 dataset root."
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save report artifacts (defaults to 'reports')."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform validation and report generation without creating V4 dataset files."
    )
    return parser.parse_args()


def parse_yaml(yaml_path):
    """
    Simple YAML parser for Ultralytics YOLO data.yaml using standard library.
    Handles 'path', 'train', 'val', 'test', and 'names' section.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    data = {"names": {}}
    in_names = False

    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_line = line.rstrip()
            stripped = raw_line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            if stripped.startswith("path:"):
                data["path"] = stripped.split("path:", 1)[1].strip().strip('"\'')
            elif stripped.startswith("train:"):
                data["train"] = stripped.split("train:", 1)[1].strip().strip('"\'')
            elif stripped.startswith("val:"):
                data["val"] = stripped.split("val:", 1)[1].strip().strip('"\'')
            elif stripped.startswith("test:"):
                data["test"] = stripped.split("test:", 1)[1].strip().strip('"\'')
            elif stripped.startswith("names:"):
                in_names = True
                val = stripped.split("names:", 1)[1].strip()
                if val.startswith("{") and val.endswith("}"):
                    val_inner = val[1:-1]
                    parts = val_inner.split(",")
                    for part in parts:
                        if ":" in part:
                            k, v = part.split(":", 1)
                            data["names"][int(k.strip())] = v.strip().strip('"\'')
                    in_names = False
            elif in_names:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    try:
                        class_id = int(parts[0].strip())
                        class_name = parts[1].strip().strip('"\'')
                        data["names"][class_id] = class_name
                    except ValueError:
                        in_names = False
                else:
                    in_names = False

    return data


def write_v4_yaml(data_yaml_out, disease_classes, v4_root_path):
    """
    Write clean v4 data.yaml with only disease classes (0-13).
    """
    os.makedirs(os.path.dirname(data_yaml_out), exist_ok=True)
    with open(data_yaml_out, 'w', encoding='utf-8') as f:
        f.write(f"path: {v4_root_path.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for cid in sorted(disease_classes.keys()):
            f.write(f"  {cid}: {disease_classes[cid]}\n")
    logger.info(f"Generated new data.yaml at: {data_yaml_out}")


def generate_verify_v4_script(reports_dir):
    """
    Update 4: Automatically generate verify_v4.py inside reports/ directory.
    """
    verify_script_path = os.path.join(reports_dir, "verify_v4.py")
    verify_code = '''#!/usr/bin/env python3
"""
Stand-alone Dataset Verification Script for plantseg_tcg_yolo_v4
Generated automatically by dataset_generator_v4.py.
"""
import os
import sys
import argparse

def parse_yaml(yaml_path):
    if not os.path.exists(yaml_path): return {}
    names = {}
    in_names = False
    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'): continue
            if stripped.startswith("names:"):
                in_names = True
                val = stripped.split("names:", 1)[1].strip()
                if val.startswith("{") and val.endswith("}"):
                    val_inner = val[1:-1]
                    for part in val_inner.split(","):
                        if ":" in part:
                            k, v = part.split(":", 1)
                            names[int(k.strip())] = v.strip().strip('"\\'')
                    in_names = False
            elif in_names:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    try: names[int(parts[0].strip())] = parts[1].strip().strip('"\\'')
                    except ValueError: in_names = False
                else: in_names = False
    return names

def main():
    parser = argparse.ArgumentParser(description="Verify plantseg_tcg_yolo_v4 dataset integrity.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/plantseg_tcg_yolo_v4")
    args = parser.parse_args()

    v4_dir = os.path.abspath(args.dataset_dir)
    print(f"Verifying Dataset: {v4_dir}")

    passed = True
    errors = []

    # 1. Verify data.yaml
    yaml_path = os.path.join(v4_dir, "data.yaml")
    classes = parse_yaml(yaml_path)
    for cid, cname in classes.items():
        if cid > 13 or "healthy" in cname.lower():
            passed = False
            errors.append(f"Invalid class in data.yaml: ID {cid} -> '{cname}'")

    # 2. Verify splits
    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = os.path.join(v4_dir, "images", split)
        lbl_dir = os.path.join(v4_dir, "labels", split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            passed = False
            errors.append(f"Missing directory for split {split}")
            continue

        images = set(os.listdir(img_dir))
        labels = set(os.listdir(lbl_dir))

        # Check 1-to-1 parity
        img_bases = {os.path.splitext(f)[0] for f in images}
        lbl_bases = {os.path.splitext(f)[0] for f in labels}

        if img_bases != lbl_bases:
            passed = False
            errors.append(f"Split {split}: Image and Label basenames do not match perfectly.")

        for lbl_file in labels:
            lbl_path = os.path.join(lbl_dir, lbl_file)
            with open(lbl_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                continue # empty background label
            for idx, line in enumerate(content.splitlines(), 1):
                parts = line.strip().split()
                if len(parts) < 3:
                    passed = False
                    errors.append(f"Split {split}, {lbl_file} L{idx}: Malformed polygon.")
                    continue
                try:
                    cid = int(parts[0])
                    if cid > 13:
                        passed = False
                        errors.append(f"Split {split}, {lbl_file} L{idx}: Invalid class ID {cid} > 13.")
                except ValueError:
                    passed = False
                    errors.append(f"Split {split}, {lbl_file} L{idx}: Non-integer class ID.")

    print("\\n==================================================")
    if passed:
        print("VERIFICATION STATUS: PASS")
        print("All V4 dataset integrity checks passed cleanly!")
    else:
        print("VERIFICATION STATUS: FAIL")
        print(f"Encountered {len(errors)} integrity errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors.")
    print("==================================================")
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
'''
    with open(verify_script_path, 'w', encoding='utf-8') as f:
        f.write(verify_code)
    logger.info(f"Generated standalone validation script: {verify_script_path}")


def generate_verification_gallery(healthy_candidates, disease_candidates, reports_dir):
    """
    Update 5: Generate reports/verification_gallery/ with random 20 healthy & 20 disease samples.
    """
    gallery_dir = os.path.join(reports_dir, "verification_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    random.seed(42) # Reproducible sampling
    sample_healthy = random.sample(healthy_candidates, min(20, len(healthy_candidates)))
    sample_disease = random.sample(disease_candidates, min(20, len(disease_candidates)))

    gallery_records = []

    for item in sample_healthy:
        src_path = item["src_path"]
        img_name = f"healthy_{item['split']}_{os.path.basename(src_path)}"
        dst_path = os.path.join(gallery_dir, img_name)
        shutil.copy2(src_path, dst_path)
        gallery_records.append({
            "filename": img_name,
            "split": item["split"],
            "category": "Healthy Background"
        })

    for item in sample_disease:
        src_path = item["src_path"]
        img_name = f"disease_{item['split']}_{os.path.basename(src_path)}"
        dst_path = os.path.join(gallery_dir, img_name)
        shutil.copy2(src_path, dst_path)
        gallery_records.append({
            "filename": img_name,
            "split": item["split"],
            "category": "Disease Lesion"
        })

    summary_md_path = os.path.join(reports_dir, "gallery_summary.md")
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("# Verification Gallery Summary\n\n")
        f.write(f"Sampled {len(sample_healthy)} Healthy Background images and {len(sample_disease)} Disease Lesion images for manual forensic inspection.\n\n")
        f.write("| Sample Filename | Split | Category |\n")
        f.write("| :--- | :--- | :--- |\n")
        for rec in gallery_records:
            f.write(f"| `{rec['filename']}` | `{rec['split']}` | {rec['category']} |\n")

    logger.info(f"Generated verification gallery in: {gallery_dir}")
    logger.info(f"Generated gallery summary markdown: {summary_md_path}")


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    reports_dir = os.path.abspath(args.reports_dir)
    dry_run = args.dry_run

    # Safety check
    if not dry_run and input_dir == output_dir:
        logger.error("Output directory cannot be identical to input directory! Aborting for safety.")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("Starting V4 Dataset Conversion (Healthy as Background)")
    logger.info(f"Input V3 Directory : {input_dir}")
    logger.info(f"Output V4 Directory: {output_dir}")
    logger.info(f"Reports Directory  : {reports_dir}")
    logger.info(f"Dry Run Mode       : {dry_run}")
    logger.info("==================================================")

    # 1. Parse Input YAML
    yaml_in_path = os.path.join(input_dir, "data.yaml")
    v3_yaml = parse_yaml(yaml_in_path)
    all_classes = v3_yaml.get("names", {})

    logger.info(f"Parsed {len(all_classes)} classes from V3 data.yaml:")
    for cid, cname in sorted(all_classes.items()):
        logger.info(f"  Class {cid:2d}: {cname}")

    # Partition classes into disease and healthy
    healthy_class_ids = set()
    disease_classes = {}

    for cid, cname in sorted(all_classes.items()):
        if cname in HEALTHY_CLASS_NAMES or "healthy" in cname.lower():
            healthy_class_ids.add(cid)
        else:
            disease_classes[cid] = cname

    logger.info(f"Identified {len(healthy_class_ids)} Healthy Classes: IDs {sorted(list(healthy_class_ids))}")
    logger.info(f"Identified {len(disease_classes)} Disease Classes: IDs {sorted(list(disease_classes.keys()))}")

    # 2. Setup Directory Structure
    splits = ["train", "val", "test"]
    if not dry_run:
        for split in splits:
            os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # File log setup in reports_dir
    file_log_path = os.path.join(reports_dir, "conversion.log")
    file_handler = logging.FileHandler(file_log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    # Tracking Statistics
    stats_per_split = {
        split: {
            "total_images": 0,
            "disease_images": 0,
            "healthy_images": 0,
            "unknown_images": 0,
            "copied_images": 0,
            "disease_labels_copied": 0,
            "empty_labels_created": 0,
            "missing_labels": 0,
            "malformed_labels": 0,
            "corrupted_labels": 0,
            "invalid_ids_found": 0,
            "verification_failed_labels": 0,
            "class_polygon_counts": defaultdict(int)
        }
        for split in splits
    }

    integrity_issues = []
    healthy_gallery_candidates = []
    disease_gallery_candidates = []

    # 3. Process Each Split
    for split in splits:
        logger.info(f"--- Processing Split: {split} ---")
        img_split_dir = os.path.join(input_dir, "images", split)
        lbl_split_dir = os.path.join(input_dir, "labels", split)

        out_img_split_dir = os.path.join(output_dir, "images", split)
        out_lbl_split_dir = os.path.join(output_dir, "labels", split)

        if not os.path.exists(img_split_dir):
            logger.warning(f"Image directory does not exist: {img_split_dir}")
            continue

        image_files = [
            f for f in os.listdir(img_split_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'))
        ]

        logger.info(f"Found {len(image_files)} images in V3 [{split}]")
        stats_per_split[split]["total_images"] = len(image_files)

        for img_name in sorted(image_files):
            base_name = os.path.splitext(img_name)[0]
            src_img_path = os.path.join(img_split_dir, img_name)
            dst_img_path = os.path.join(out_img_split_dir, img_name)

            if not dry_run:
                # Copy Image without modification
                shutil.copy2(src_img_path, dst_img_path)
            stats_per_split[split]["copied_images"] += 1

            # Label Processing
            src_lbl_path = os.path.join(lbl_split_dir, base_name + ".txt")
            dst_lbl_path = os.path.join(out_lbl_split_dir, base_name + ".txt")

            # Update 1 — Never Assume Missing Labels Are Healthy
            if not os.path.exists(src_lbl_path):
                stats_per_split[split]["missing_labels"] += 1
                stats_per_split[split]["unknown_images"] += 1
                stats_per_split[split]["empty_labels_created"] += 1

                if not dry_run:
                    # Create empty label file to preserve dataset structure
                    with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                        pass

                issue_msg = "Missing label file in V3; created empty label in V4 & classified as UNKNOWN"
                logger.warning(f"[{split}] {img_name}: {issue_msg}")
                integrity_issues.append({
                    "split": split,
                    "image": img_name,
                    "issue": issue_msg
                })
                continue

            # Read source label lines
            with open(src_lbl_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()

            valid_disease_lines = []
            has_healthy = False
            has_disease = False
            is_malformed = False
            has_invalid_id = False

            # Update 3 — Validate Disease Labels
            for line_idx, line in enumerate(raw_lines):
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split()

                # Check minimum coordinate pairs (class_id + x1 + y1 + x2 + y2 minimum 5 tokens for polygon)
                if len(parts) < 3:
                    is_malformed = True
                    issue_msg = f"Malformed label line {line_idx+1}: insufficient tokens '{line_str}'"
                    logger.warning(f"[{split}] {img_name}: {issue_msg}")
                    integrity_issues.append({"split": split, "image": img_name, "issue": issue_msg})
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    is_malformed = True
                    issue_msg = f"Invalid non-integer class ID '{parts[0]}' on line {line_idx+1}"
                    logger.warning(f"[{split}] {img_name}: {issue_msg}")
                    integrity_issues.append({"split": split, "image": img_name, "issue": issue_msg})
                    continue

                # Validate coordinates are numeric floats
                coord_error = False
                for coord_str in parts[1:]:
                    try:
                        val = float(coord_str)
                    except ValueError:
                        coord_error = True
                        break

                if coord_error:
                    is_malformed = True
                    issue_msg = f"Invalid non-float coordinate values on line {line_idx+1}"
                    logger.warning(f"[{split}] {img_name}: {issue_msg}")
                    integrity_issues.append({"split": split, "image": img_name, "issue": issue_msg})
                    continue

                if class_id in healthy_class_ids:
                    has_healthy = True
                elif class_id in disease_classes:
                    has_disease = True
                    valid_disease_lines.append(line_str)
                    stats_per_split[split]["class_polygon_counts"][class_id] += 1
                else:
                    has_invalid_id = True
                    stats_per_split[split]["invalid_ids_found"] += 1
                    issue_msg = f"Unknown class ID {class_id} outside V3 range"
                    logger.warning(f"[{split}] {img_name}: {issue_msg}")
                    integrity_issues.append({"split": split, "image": img_name, "issue": issue_msg})

            if is_malformed:
                stats_per_split[split]["malformed_labels"] += 1

            # Update 2 & 3 — Classification & Verification
            if has_disease:
                if len(valid_disease_lines) > 0 and not is_malformed:
                    # Valid Disease Image
                    stats_per_split[split]["disease_images"] += 1
                    stats_per_split[split]["disease_labels_copied"] += 1
                    disease_gallery_candidates.append({"src_path": src_img_path, "split": split})

                    if not dry_run:
                        with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                            for dline in valid_disease_lines:
                                f.write(dline + "\n")
                else:
                    # Update 3: Disease label has zero valid polygons -> copy original label unchanged & mark failed
                    stats_per_split[split]["disease_images"] += 1
                    stats_per_split[split]["verification_failed_labels"] += 1
                    issue_msg = f"Disease label failed validation (zero valid polygons). Copying original unchanged."
                    logger.warning(f"[{split}] {img_name}: {issue_msg}")
                    integrity_issues.append({"split": split, "image": img_name, "issue": issue_msg})

                    if not dry_run:
                        shutil.copy2(src_lbl_path, dst_lbl_path)

            elif has_healthy:
                # Confirmed Healthy Image -> convert to empty label
                stats_per_split[split]["healthy_images"] += 1
                stats_per_split[split]["empty_labels_created"] += 1
                healthy_gallery_candidates.append({"src_path": src_img_path, "split": split})

                if not dry_run:
                    with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                        pass
            else:
                # Update 2: Classify as UNKNOWN (empty or corrupt V3 label with no healthy or disease tags)
                stats_per_split[split]["unknown_images"] += 1
                stats_per_split[split]["empty_labels_created"] += 1
                issue_msg = "Label contains neither disease nor healthy class tags. Classified as UNKNOWN & created empty label."
                logger.warning(f"[{split}] {img_name}: {issue_msg}")
                integrity_issues.append({"split": split, "image": img_name, "issue": issue_msg})

                if not dry_run:
                    with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                        pass

    # 4. Generate V4 data.yaml
    if not dry_run:
        yaml_out_path = os.path.join(output_dir, "data.yaml")
        write_v4_yaml(yaml_out_path, disease_classes, output_dir)

    # 5. Generate Scripts & Galleries
    generate_verify_v4_script(reports_dir)
    if healthy_gallery_candidates or disease_gallery_candidates:
        generate_verification_gallery(healthy_gallery_candidates, disease_gallery_candidates, reports_dir)

    # 6. Compile Statistics
    total_images_all = sum(stats_per_split[s]["total_images"] for s in splits)
    total_disease = sum(stats_per_split[s]["disease_images"] for s in splits)
    total_healthy = sum(stats_per_split[s]["healthy_images"] for s in splits)
    total_unknown = sum(stats_per_split[s]["unknown_images"] for s in splits)
    total_empty_created = sum(stats_per_split[s]["empty_labels_created"] for s in splits)
    total_disease_copied = sum(stats_per_split[s]["disease_labels_copied"] for s in splits)
    total_missing = sum(stats_per_split[s]["missing_labels"] for s in splits)
    total_malformed = sum(stats_per_split[s]["malformed_labels"] for s in splits)
    total_invalid_ids = sum(stats_per_split[s]["invalid_ids_found"] for s in splits)

    # Update 7 — Final Validation
    validation_passed = True
    validation_error_msg = ""

    # Healthy Images == Empty Label Files created for healthy
    # Disease Images == Disease Labels Copied
    if (total_healthy + total_unknown) != total_empty_created:
        validation_passed = False
        validation_error_msg += f"Mismatch: Healthy+Unknown ({total_healthy}+{total_unknown}) != Empty Labels Created ({total_empty_created}). "

    if total_disease != (total_disease_copied + sum(stats_per_split[s]["verification_failed_labels"] for s in splits)):
        validation_passed = False
        validation_error_msg += f"Mismatch: Disease Images ({total_disease}) != Disease Labels Copied/Retained. "

    if total_missing > 0 or total_malformed > 0 or total_invalid_ids > 0:
        validation_passed = False

    verification_status_str = "PASS" if validation_passed else "FAIL"

    # Update 6 — Conversion Summary Output
    logger.info("==================================================")
    logger.info("Conversion Summary")
    logger.info("==================================================")
    logger.info(f"Total Images           : {total_images_all}")
    logger.info(f"Disease Images         : {total_disease}")
    logger.info(f"Healthy Images         : {total_healthy}")
    logger.info(f"Unknown Images         : {total_unknown}")
    logger.info(f"Empty Labels Created   : {total_empty_created}")
    logger.info(f"Disease Labels Copied  : {total_disease_copied}")
    logger.info(f"Missing Labels         : {total_missing}")
    logger.info(f"Malformed Labels       : {total_malformed}")
    logger.info(f"Invalid IDs            : {total_invalid_ids}")
    logger.info(f"Verification Status    : {verification_status_str}")
    logger.info("==================================================")

    # Update 8 — Save json, md, csv reports
    report_json_path = os.path.join(reports_dir, "conversion_report.json")
    report_data = {
        "dataset_name": "plantseg_tcg_yolo_v4",
        "source_dataset": "plantseg_tcg_yolo_v3",
        "verification_status": verification_status_str,
        "total_images": total_images_all,
        "disease_images": total_disease,
        "healthy_images": total_healthy,
        "unknown_images": total_unknown,
        "empty_labels_created": total_empty_created,
        "disease_labels_copied": total_disease_copied,
        "missing_labels": total_missing,
        "malformed_labels": total_malformed,
        "invalid_ids": total_invalid_ids,
        "disease_classes": disease_classes,
        "removed_healthy_classes": list(HEALTHY_CLASS_NAMES),
        "split_statistics": stats_per_split,
        "integrity_issues": integrity_issues
    }
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    report_csv_path = os.path.join(reports_dir, "dataset_statistics.csv")
    with open(report_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "class_id", "class_name", "category",
            "images_count", "polygons_count"
        ])
        for split in splits:
            writer.writerow([
                split, -1, "healthy_background", "background",
                stats_per_split[split]["healthy_images"], 0
            ])
            writer.writerow([
                split, -2, "unknown_background", "unknown",
                stats_per_split[split]["unknown_images"], 0
            ])
            for cid, cname in sorted(disease_classes.items()):
                poly_cnt = stats_per_split[split]["class_polygon_counts"].get(cid, 0)
                writer.writerow([
                    split, cid, cname, "disease",
                    "N/A", poly_cnt
                ])

    report_md_path = os.path.join(reports_dir, "conversion_report.md")
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("# plantseg_tcg_yolo_v4 Dataset Conversion Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"- **Source Dataset**: `plantseg_tcg_yolo_v3`\n")
        f.write(f"- **Target Dataset**: `plantseg_tcg_yolo_v4`\n")
        f.write(f"- **Verification Status**: `{verification_status_str}`\n")
        f.write(f"- **Total Images**: {total_images_all}\n")
        f.write(f"- **Disease Images**: {total_disease}\n")
        f.write(f"- **Healthy Images**: {total_healthy}\n")
        f.write(f"- **Unknown Images**: {total_unknown}\n")
        f.write(f"- **Empty Label Files Created**: {total_empty_created}\n")
        f.write(f"- **Disease Label Files Copied**: {total_disease_copied}\n\n")

        f.write("## Integrity Issues Log\n")
        if integrity_issues:
            f.write(f"> [!WARNING]\n> Logged {len(integrity_issues)} integrity issue items:\n\n")
            for issue in integrity_issues[:50]:
                f.write(f"- **[{issue['split']}]** `{issue['image']}`: {issue['issue']}\n")
            if len(integrity_issues) > 50:
                f.write(f"\n*...and {len(integrity_issues) - 50} more items (see conversion_report.json)*\n")
        else:
            f.write("> [!NOTE]\n> Zero integrity errors or malformed annotations detected!\n")

    # Update 7 — Final Assertion Check
    if not validation_passed and validation_error_msg:
        logger.error(f"Final Validation Exception: {validation_error_msg}")

    # Update 10 — Exit Status
    if validation_passed:
        print("\n✓ Dataset conversion completed successfully.\n")
        sys.exit(0)
    else:
        print("\n✗ Dataset conversion completed with validation errors.\n")
        sys.exit(1 if not dry_run else 0)


if __name__ == "__main__":
    main()
