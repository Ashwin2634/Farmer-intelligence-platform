import os
import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Map original class IDs to information about them.
# The keys are original class IDs.
# For each original class:
# - new_name: the snake_case standardized name
# - action: "excluded", "kept", "merged"
ORIGINAL_CLASSES = {
    0: {"name": "12", "new_name": None, "action": "excluded"},
    1: {"name": "Apple Scab Leaf", "new_name": "apple_scab", "action": "kept"},
    2: {"name": "Apple leaf", "new_name": "apple_healthy", "action": "kept"},
    3: {"name": "Apple rust leaf", "new_name": "apple_rust", "action": "kept"},
    4: {"name": "Bell_pepper leaf", "new_name": "bell_pepper_healthy", "action": "kept"},
    5: {"name": "Bell_pepper leaf spot", "new_name": "bell_pepper_leaf_spot", "action": "kept"},
    6: {"name": "Blueberry leaf", "new_name": "blueberry_healthy", "action": "kept"},
    7: {"name": "Cherry leaf", "new_name": "cherry_healthy", "action": "kept"},
    8: {"name": "Corn Gray leaf spot", "new_name": "corn_gray_leaf_spot", "action": "kept"},
    9: {"name": "Corn leaf blight", "new_name": "corn_leaf_blight", "action": "kept"},
    10: {"name": "Corn rust leaf", "new_name": "corn_rust", "action": "kept"},
    11: {"name": "Peach leaf", "new_name": "peach_healthy", "action": "kept"},
    12: {"name": "Potato leaf", "new_name": "potato_healthy", "action": "kept"},
    13: {"name": "Potato leaf early blight", "new_name": None, "action": "excluded"},
    14: {"name": "Potato leaf late blight", "new_name": "potato_late_blight", "action": "kept"},
    15: {"name": "Raspberry leaf", "new_name": "raspberry_healthy", "action": "kept"},
    16: {"name": "Soyabean leaf", "new_name": "soybean_healthy", "action": "merged"},
    17: {"name": "Soybean leaf", "new_name": "soybean_healthy", "action": "merged"},
    18: {"name": "Squash Powdery mildew leaf", "new_name": "squash_powdery_mildew", "action": "kept"},
    19: {"name": "Strawberry leaf", "new_name": "strawberry_healthy", "action": "kept"},
    20: {"name": "Tomato Early blight leaf", "new_name": "tomato_early_blight", "action": "kept"},
    21: {"name": "Tomato Septoria leaf spot", "new_name": "tomato_septoria_leaf_spot", "action": "kept"},
    22: {"name": "Tomato leaf", "new_name": "tomato_healthy", "action": "kept"},
    23: {"name": "Tomato leaf bacterial spot", "new_name": "tomato_bacterial_spot", "action": "kept"},
    24: {"name": "Tomato leaf late blight", "new_name": "tomato_late_blight", "action": "kept"},
    25: {"name": "Tomato leaf mosaic virus", "new_name": "tomato_mosaic_virus", "action": "kept"},
    26: {"name": "Tomato leaf yellow virus", "new_name": "tomato_yellow_virus", "action": "kept"},
    27: {"name": "Tomato mold leaf", "new_name": "tomato_leaf_mold", "action": "kept"},
    28: {"name": "Tomato two spotted spider mites leaf", "new_name": None, "action": "excluded"},
    29: {"name": "grape leaf", "new_name": "grape_healthy", "action": "kept"},
    30: {"name": "grape leaf black rot", "new_name": "grape_black_rot", "action": "kept"},
}

def generate_mappings() -> Tuple[Dict[int, int], List[str], Dict[int, Dict[str, Any]]]:
    """
    Generates class mappings ensuring continuous new class IDs starting from 0.
    """
    new_class_names = []
    name_to_new_id = {}
    
    # First, collect all unique kept/merged class names in order
    for orig_id in sorted(ORIGINAL_CLASSES.keys()):
        info = ORIGINAL_CLASSES[orig_id]
        if info["action"] in ("kept", "merged"):
            new_name = info["new_name"]
            if new_name not in name_to_new_id:
                name_to_new_id[new_name] = len(new_class_names)
                new_class_names.append(new_name)
                
    # Now build the direct remapping dict
    id_mapping = {}
    mapping_metadata = {}
    
    for orig_id in sorted(ORIGINAL_CLASSES.keys()):
        info = ORIGINAL_CLASSES[orig_id]
        action = info["action"]
        orig_name = info["name"]
        
        if action == "excluded":
            id_mapping[orig_id] = None
            mapping_metadata[orig_id] = {
                "original_class_id": orig_id,
                "original_class_name": orig_name,
                "new_class_id": None,
                "new_class_name": None,
                "action": "excluded"
            }
        else:
            new_name = info["new_name"]
            new_id = name_to_new_id[new_name]
            id_mapping[orig_id] = new_id
            mapping_metadata[orig_id] = {
                "original_class_id": orig_id,
                "original_class_name": orig_name,
                "new_class_id": new_id,
                "new_class_name": new_name,
                "action": "renamed" if action == "kept" else "merged"
            }
            
    return id_mapping, new_class_names, mapping_metadata

def compute_file_hash(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()

def detect_leakage(src_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Scans train, valid, test splits to detect exact duplicate image files across them.
    """
    splits = ["train", "valid", "test"]
    hash_map = {} # hash -> list of (split, filename, path)
    
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    
    for split in splits:
        img_dir = src_dir / split / "images"
        if not img_dir.exists():
            continue
        for file_path in img_dir.iterdir():
            if file_path.suffix in img_extensions:
                file_hash = compute_file_hash(file_path)
                if file_hash not in hash_map:
                    hash_map[file_hash] = []
                hash_map[file_hash].append({
                    "split": split,
                    "filename": file_path.name,
                    "path": str(file_path)
                })
                
    # Filter only duplicates that cross split boundaries
    leakage = {}
    for file_hash, occurrences in hash_map.items():
        if len(occurrences) > 1:
            # Check if there's more than one distinct split
            distinct_splits = {occ["split"] for occ in occurrences}
            if len(distinct_splits) > 1:
                leakage[file_hash] = occurrences
                
    return leakage

def clean_dataset(src_dir: Path, dest_dir: Path, dry_run: bool = False, overwrite: bool = False):
    src_dir = Path(src_dir).resolve()
    dest_dir = Path(dest_dir).resolve()
    
    print(f"Source dataset: {src_dir}")
    print(f"Target cleaned dataset: {dest_dir}")
    
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist.")
        sys.exit(1)
        
    if dest_dir.exists() and not dry_run:
        if not overwrite:
            print(f"Error: Target directory {dest_dir} already exists. Use --overwrite to overwrite.")
            sys.exit(1)
        else:
            print(f"Warning: --overwrite specified. Target directory {dest_dir} will be deleted and recreated.")
            
    # Compute class mappings
    id_mapping, new_class_names, mapping_metadata = generate_mappings()
    
    # Check for leakage in source
    print("\nRunning data leakage check (cross-split duplicate images)...")
    leakage = detect_leakage(src_dir)
    if leakage:
        print(f"WARNING: Detected {len(leakage)} duplicate image files across different splits!")
        for file_hash, occs in list(leakage.items())[:5]:
            print(f"  Hash {file_hash[:12]}:")
            for occ in occs:
                print(f"    - split: {occ['split']}, file: {occ['filename']}")
        if len(leakage) > 5:
            print(f"    ... and {len(leakage) - 5} more.")
    else:
        print("No cross-split duplicate images detected. Clean!")
        
    # Stats trackers
    original_stats = {"images": 0, "boxes": 0, "class_counts": {}}
    cleaned_stats = {"images": 0, "boxes": 0, "class_counts": {i: 0 for i in range(len(new_class_names))}}
    removed_counts = {0: 0, 13: 0, 28: 0} # stats for excluded classes
    
    splits = ["train", "valid", "test"]
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    
    # In dry-run, we do not write to disk
    if not dry_run:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Save class mapping JSON
        class_mapping_path = dest_dir / "class_mapping.json"
        with open(class_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(list(mapping_metadata.values()), f, indent=2)
        print(f"Saved class mapping JSON to: {class_mapping_path}")

    # Process splits
    for split in splits:
        src_split_img = src_dir / split / "images"
        src_split_lbl = src_dir / split / "labels"
        
        dest_split_img = dest_dir / split / "images"
        dest_split_lbl = dest_dir / split / "labels"
        
        if not src_split_img.exists():
            print(f"Warning: split directory {src_split_img} not found, skipping split {split}.")
            continue
            
        if not dry_run:
            dest_split_img.mkdir(parents=True, exist_ok=True)
            dest_split_lbl.mkdir(parents=True, exist_ok=True)
            
        # List files
        img_files = [p for p in src_split_img.iterdir() if p.suffix in img_extensions]
        
        print(f"\nProcessing split '{split}': {len(img_files)} images found.")
        
        for img_path in img_files:
            original_stats["images"] += 1
            lbl_path = src_split_lbl / f"{img_path.stem}.txt"
            
            # Read and filter annotations
            valid_new_lines = []
            img_has_boxes = False
            
            if lbl_path.exists():
                try:
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        lines = f.read().splitlines()
                except Exception as e:
                    print(f"Error reading annotation {lbl_path}: {e}")
                    lines = []
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        # Malformed, preserve as is or raise?
                        # Requirements say: "Never silently ignore malformed annotations."
                        raise ValueError(f"Malformed annotation in {lbl_path}: '{line}'")
                        
                    orig_cls_id = int(parts[0])
                    coords = parts[1:]
                    
                    original_stats["boxes"] += 1
                    original_stats["class_counts"][orig_cls_id] = original_stats["class_counts"].get(orig_cls_id, 0) + 1
                    
                    new_cls_id = id_mapping.get(orig_cls_id)
                    if new_cls_id is None:
                        # Excluded class
                        if orig_cls_id in removed_counts:
                            removed_counts[orig_cls_id] += 1
                        else:
                            removed_counts[orig_cls_id] = removed_counts.get(orig_cls_id, 0) + 1
                    else:
                        # Kept or merged
                        valid_new_lines.append(f"{new_cls_id} {' '.join(coords)}")
                        cleaned_stats["boxes"] += 1
                        cleaned_stats["class_counts"][new_cls_id] += 1
                        img_has_boxes = True
            
            # Decide on image copy and background labeling
            # YOLO background images are represented by an empty label file
            cleaned_stats["images"] += 1
            
            if not dry_run:
                # Copy image
                shutil.copy2(img_path, dest_split_img / img_path.name)
                # Write labels
                new_lbl_path = dest_split_lbl / f"{img_path.stem}.txt"
                with open(new_lbl_path, 'w', encoding='utf-8') as f:
                    if valid_new_lines:
                        f.write('\n'.join(valid_new_lines) + '\n')
                        
    # Write clean data.yaml
    if not dry_run:
        clean_yaml = {
            "path": str(dest_dir),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(new_class_names),
            "names": new_class_names
        }
        yaml_out_path = dest_dir / "data.yaml"
        with open(yaml_out_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {clean_yaml['path']}\n")
            f.write(f"train: {clean_yaml['train']}\n")
            f.write(f"val: {clean_yaml['val']}\n")
            f.write(f"test: {clean_yaml['test']}\n\n")
            f.write(f"nc: {clean_yaml['nc']}\n")
            f.write(f"names: {clean_yaml['names']}\n")
            
        print(f"\nGenerated cleaned dataset YAML: {yaml_out_path}")
        
    # Compile reports
    report_data = {
        "dry_run": dry_run,
        "original_dataset": str(src_dir),
        "cleaned_dataset": str(dest_dir),
        "classes_excluded": {
            "0 (12)": removed_counts.get(0, 0),
            "13 (Potato Early Blight)": removed_counts.get(13, 0),
            "28 (Tomato Spider Mites)": removed_counts.get(28, 0),
        },
        "classes_merged": "Soyabean (16) + Soybean (17) -> soybean_healthy",
        "original_stats": {
            "classes": len(ORIGINAL_CLASSES),
            "images": original_stats["images"],
            "bounding_boxes": original_stats["boxes"]
        },
        "cleaned_stats": {
            "classes": len(new_class_names),
            "images": cleaned_stats["images"],
            "bounding_boxes": cleaned_stats["boxes"]
        },
        "background_images_strategy": "Preserved empty label file for images with only excluded classes (Ultralytics YOLO standard)",
        "data_leakage": {
            "duplicate_cross_split_hashes_count": len(leakage),
            "duplicate_files": {h: [{"split": o["split"], "filename": o["filename"]} for o in occs] for h, occs in leakage.items()}
        }
    }
    
    # Save clean dataset report
    if not dry_run:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / "clean_dataset_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved clean report to: {report_path}")
        
    # Print before vs after summary
    print("\n" + "="*50)
    print("CLEANING BEFORE-VS-AFTER SUMMARY")
    print("="*50)
    print(f"Original:")
    print(f"  Classes: {len(ORIGINAL_CLASSES)}")
    print(f"  Images: {original_stats['images']}")
    print(f"  Bounding Boxes: {original_stats['boxes']}")
    print(f"\nCleaned:")
    print(f"  Classes: {len(new_class_names)}")
    print(f"  Images: {cleaned_stats['images']}")
    print(f"  Bounding Boxes: {cleaned_stats['boxes']}")
    print(f"\nRemoved Annotations:")
    print(f"  ID 0 (12): {removed_counts.get(0, 0)}")
    print(f"  ID 13 (Potato early blight): {removed_counts.get(13, 0)}")
    print(f"  ID 28 (Tomato spider mites): {removed_counts.get(28, 0)}")
    print(f"\nMerged:")
    print(f"  Soyabean + Soybean -> soybean_healthy (remapped to ID 14)")
    print(f"\nBackground image strategy: {report_data['background_images_strategy']}")
    if dry_run:
        print("\n*** DRY RUN MODE: No files were written or modified ***")
    else:
        print("\nCleaning process completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Clean and standardize plant disease YOLO dataset.")
    parser.add_argument("--src", type=str, default="datasets/plant_disease_v3", help="Source dataset path")
    parser.add_argument("--dest", type=str, default="datasets/plant_disease_clean", help="Destination clean dataset path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without writing to destination")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination folder if it exists")
    args = parser.parse_args()
    
    clean_dataset(Path(args.src), Path(args.dest), dry_run=args.dry_run, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
