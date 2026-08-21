import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Try importing PIL to check image readability
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def parse_simple_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Parses a YOLO data.yaml file without requiring external libraries.
    Handles simple structures like key: value and lists.
    """
    data = {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple line-by-line parsing
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            
            # Handle inline list e.g. names: ['12', 'Apple Scab Leaf', ...]
            if val.startswith('[') and val.endswith(']'):
                # Extract items
                list_str = val[1:-1]
                # split by comma, clean quotes
                items = []
                for x in list_str.split(','):
                    x = x.strip()
                    if (x.startswith("'") and x.endswith("'")) or (x.startswith('"') and x.endswith('"')):
                        items.append(x[1:-1])
                    else:
                        items.append(x)
                data[key] = items
            # Handle multiline list or nested block
            elif not val:
                # check next lines for indentation or list elements
                j = i + 1
                list_items = []
                is_list = False
                while j < len(lines):
                    next_line = lines[j]
                    if not next_line.strip():
                        j += 1
                        continue
                    if not next_line.startswith(' ') and not next_line.startswith('-'):
                        break
                    
                    nl_strip = next_line.strip()
                    if nl_strip.startswith('-'):
                        is_list = True
                        item = nl_strip[1:].strip()
                        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                            item = item[1:-1]
                        list_items.append(item)
                    elif nl_strip.startswith("'") or nl_strip.startswith('"') or ',' in nl_strip:
                        # might be part of names: [ ... ] on multiple lines
                        pass
                    j += 1
                
                # Check if we found names block on multiple lines
                if key == 'names' and '[' in line:
                    # Let's read until we find ']'
                    multiline_val = line.split(':', 1)[1].strip()
                    k = i + 1
                    while k < len(lines) and ']' not in multiline_val:
                        multiline_val += " " + lines[k].strip()
                        k += 1
                    multiline_val = multiline_val.strip()
                    if multiline_val.startswith('[') and multiline_val.endswith(']'):
                        list_str = multiline_val[1:-1]
                        items = []
                        # Parse elements handling quotes safely
                        # Simple tokenizer for quoted strings
                        import re
                        raw_items = re.findall(r"'(.*?)'|\"(.*?)\"", list_str)
                        items = [r[0] if r[0] else r[1] for r in raw_items]
                        data[key] = items
                        i = k - 1
                elif is_list:
                    data[key] = list_items
                    i = j - 1
            else:
                # Try to parse integer or float or boolean or string
                if val.lower() == 'true':
                    data[key] = True
                elif val.lower() == 'false':
                    data[key] = False
                else:
                    try:
                        if '.' in val:
                            data[key] = float(val)
                        else:
                            data[key] = int(val)
                    except ValueError:
                        # Keep as string and remove optional quotes
                        if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                            val = val[1:-1]
                        data[key] = val
        i += 1
    return data

def analyze_dataset(dataset_dir: Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Error: data.yaml not found in {dataset_dir}")
        sys.exit(1)
        
    print(f"Reading configuration from: {yaml_path}")
    yaml_data = parse_simple_yaml(yaml_path)
    
    nc = yaml_data.get('nc', 0)
    names = yaml_data.get('names', [])
    print(f"Dataset config specifies {nc} classes: {names}")
    
    # Splits mapping
    splits = {}
    for key in ['train', 'val', 'valid', 'test']:
        if key in yaml_data:
            path_val = yaml_data[key]
            # Resolve relative to dataset directory or yaml path
            resolved_path = (yaml_path.parent / path_val).resolve()
            if resolved_path.exists():
                splits[key] = resolved_path
            else:
                # Try relative to project directory
                proj_path = Path(path_val).resolve()
                if proj_path.exists():
                    splits[key] = proj_path
                else:
                    # Look for directory in dataset_dir directly (e.g. dataset_dir / val)
                    direct_path = dataset_dir / Path(path_val).name
                    if direct_path.exists():
                        splits[key] = direct_path
                    else:
                        # Try standard subdirectories
                        standard_name = 'valid' if key == 'val' else key
                        std_path = dataset_dir / standard_name
                        if std_path.exists():
                            splits[key] = std_path

    # If splits are images paths, we check parent or subfolders
    print("Detected splits and directory paths:")
    for split, path in splits.items():
        print(f"  - {split}: {path}")

    analysis_results = {
        "dataset_directory": str(dataset_dir),
        "yaml_config": yaml_data,
        "splits": {}
    }

    # Allowed image extensions
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}

    for split_name, split_path in splits.items():
        print(f"\nAnalyzing split: {split_name}...")
        
        # Check if the split path is a directory of images directly, or has images/ labels/ structure
        images_dir = split_path
        labels_dir = split_path.parent / 'labels'
        
        # In typical Ultralytics / Roboflow structure, the split path itself might point to 'train/images'
        # or it might be 'train' and contain 'images' and 'labels' subfolders.
        if images_dir.name == 'images':
            labels_dir = images_dir.parent / 'labels'
        elif (images_dir / 'images').exists():
            labels_dir = images_dir / 'labels'
            images_dir = images_dir / 'images'
            
        print(f"  Images Directory: {images_dir}")
        print(f"  Labels Directory: {labels_dir}")
        
        # Find all images
        image_files = []
        if images_dir.exists() and images_dir.is_dir():
            image_files = [p for p in images_dir.iterdir() if p.suffix in img_extensions]
        
        # Find all label files
        label_files = []
        if labels_dir.exists() and labels_dir.is_dir():
            label_files = [p for p in labels_dir.iterdir() if p.suffix == '.txt']
            
        image_names = {p.stem: p for p in image_files}
        label_names = {p.stem: p for p in label_files}
        
        total_images = len(image_files)
        total_labels = len(label_files)
        
        # Mismatches
        images_without_labels = []
        labels_without_images = []
        
        for stem in image_names:
            if stem not in label_names:
                images_without_labels.append(str(image_names[stem].name))
                
        for stem in label_names:
            if stem not in image_names:
                labels_without_images.append(str(label_names[stem].name))
                
        empty_label_files = []
        corrupted_images = []
        malformed_annotations = []
        out_of_range_classes = []
        invalid_coordinates = []
        out_of_bounds_boxes = []
        
        class_counts = {i: 0 for i in range(nc)}
        # Support extra counts if they exist in labels
        extra_class_counts = {}
        
        total_boxes = 0
        
        # Read and check images for corruption
        for stem, img_path in image_names.items():
            if HAS_PIL:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted_images.append(img_path.name)
            else:
                # Basic check: is size > 0
                if img_path.stat().st_size == 0:
                    corrupted_images.append(img_path.name)

        # Read and parse labels
        for stem, lbl_path in label_names.items():
            # Read label content
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
            except Exception as e:
                malformed_annotations.append(f"{lbl_path.name}: Failed to read file - {str(e)}")
                continue
                
            if not lines or all(not l.strip() for l in lines):
                empty_label_files.append(lbl_path.name)
                continue
                
            for line_idx, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    malformed_annotations.append(f"{lbl_path.name} [line {line_idx}]: Expected 5 fields, got {len(parts)} (content: '{line}')")
                    continue
                    
                # Parse class ID
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    malformed_annotations.append(f"{lbl_path.name} [line {line_idx}]: Class ID is not integer ('{parts[0]}')")
                    continue
                    
                if cls_id < 0 or cls_id >= nc:
                    out_of_range_classes.append(f"{lbl_path.name} [line {line_idx}]: Class ID {cls_id} is out of range [0, {nc-1}]")
                    extra_class_counts[cls_id] = extra_class_counts.get(cls_id, 0) + 1
                else:
                    class_counts[cls_id] += 1
                    
                # Parse coordinates
                coords_ok = True
                coords = []
                for coord_idx, coord_str in enumerate(parts[1:], 1):
                    try:
                        val = float(coord_str)
                        coords.append(val)
                    except ValueError:
                        malformed_annotations.append(f"{lbl_path.name} [line {line_idx}]: Bounding box coordinate is not float ('{coord_str}')")
                        coords_ok = False
                        break
                
                if not coords_ok:
                    continue
                    
                x_center, y_center, width, height = coords
                
                # Validate normalized coordinates range
                if not (0.0 <= x_center <= 1.0) or not (0.0 <= y_center <= 1.0) or not (0.0 < width <= 1.0) or not (0.0 < height <= 1.0):
                    invalid_coordinates.append(
                        f"{lbl_path.name} [line {line_idx}]: Coordinates out of range: x={x_center}, y={y_center}, w={width}, h={height}"
                    )
                    
                # Bounding box boundaries check
                x1 = x_center - width / 2.0
                y1 = y_center - height / 2.0
                x2 = x_center + width / 2.0
                y2 = y_center + height / 2.0
                
                if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
                    out_of_bounds_boxes.append(
                        f"{lbl_path.name} [line {line_idx}]: Box extends outside boundaries: [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]"
                    )
                    
                total_boxes += 1
                
        # Split level summary
        split_summary = {
            "images_count": total_images,
            "labels_count": total_labels,
            "total_bounding_boxes": total_boxes,
            "images_without_labels_count": len(images_without_labels),
            "images_without_labels_samples": images_without_labels[:10],
            "labels_without_images_count": len(labels_without_images),
            "labels_without_images_samples": labels_without_images[:10],
            "empty_label_files_count": len(empty_label_files),
            "empty_label_files_samples": empty_label_files[:10],
            "corrupted_images_count": len(corrupted_images),
            "corrupted_images_samples": corrupted_images[:10],
            "malformed_annotations_count": len(malformed_annotations),
            "malformed_annotations_samples": malformed_annotations[:10],
            "out_of_range_classes_count": len(out_of_range_classes),
            "out_of_range_classes_samples": out_of_range_classes[:10],
            "invalid_coordinates_count": len(invalid_coordinates),
            "invalid_coordinates_samples": invalid_coordinates[:10],
            "out_of_bounds_boxes_count": len(out_of_bounds_boxes),
            "out_of_bounds_boxes_samples": out_of_bounds_boxes[:10],
            "class_counts": {names[cid] if cid < len(names) else f"unknown_class_{cid}": count for cid, count in class_counts.items()},
            "extra_class_counts": extra_class_counts
        }
        
        analysis_results["splits"][split_name] = split_summary

    # Overall dataset metrics checks
    # Identify classes in validation/test but 0 in train
    all_classes_in_train = set()
    train_split = analysis_results["splits"].get("train")
    if train_split:
        for cls_name, count in train_split["class_counts"].items():
            if count > 0:
                all_classes_in_train.add(cls_name)
                
    zero_train_classes = {}
    low_train_classes = {}
    
    for split_name, split_data in analysis_results["splits"].items():
        if split_name == "train":
            # Just flag low representation in train split
            for cls_name, count in split_data["class_counts"].items():
                if 0 < count < 10:
                    low_train_classes[cls_name] = count
            continue
            
        for cls_name, count in split_data["class_counts"].items():
            if count > 0 and cls_name not in all_classes_in_train:
                if cls_name not in zero_train_classes:
                    zero_train_classes[cls_name] = []
                zero_train_classes[cls_name].append(split_name)

    analysis_results["quality_issues"] = {
        "zero_train_classes": zero_train_classes,
        "low_train_classes": low_train_classes
    }

    # Print summary report
    print("\n" + "="*50)
    print("DATASET ANALYSIS SUMMARY")
    print("="*50)
    
    for split_name, split_data in analysis_results["splits"].items():
        print(f"\nSplit: {split_name.upper()}")
        print(f"  Images: {split_data['images_count']}")
        print(f"  Labels: {split_data['labels_count']}")
        print(f"  Bounding Boxes: {split_data['total_bounding_boxes']}")
        if split_data['images_without_labels_count'] > 0:
            print(f"  * Warning: Images without label files: {split_data['images_without_labels_count']}")
        if split_data['labels_without_images_count'] > 0:
            print(f"  * Error: Label files without matching image files: {split_data['labels_without_images_count']}")
        if split_data['empty_label_files_count'] > 0:
            print(f"  * Note: Empty label files: {split_data['empty_label_files_count']}")
        if split_data['corrupted_images_count'] > 0:
            print(f"  * Error: Corrupted images: {split_data['corrupted_images_count']}")
        if split_data['malformed_annotations_count'] > 0:
            print(f"  * Error: Malformed annotations: {split_data['malformed_annotations_count']}")
        if split_data['out_of_range_classes_count'] > 0:
            print(f"  * Error: Bounding boxes with out-of-range class IDs: {split_data['out_of_range_classes_count']}")
        if split_data['invalid_coordinates_count'] > 0:
            print(f"  * Error: Invalid coordinates (outside [0, 1] range): {split_data['invalid_coordinates_count']}")
        if split_data['out_of_bounds_boxes_count'] > 0:
            print(f"  * Note: Bounding boxes extending outside image limits: {split_data['out_of_bounds_boxes_count']}")

        print("  Class Distribution:")
        for cls_name, count in split_data["class_counts"].items():
            if count > 0:
                print(f"    - {cls_name}: {count}")
                
    if zero_train_classes:
        print("\nCRITICAL QUALITY WARNING: Classes in valid/test split but missing in train split:")
        for cls_name, splits_found in zero_train_classes.items():
            print(f"  - {cls_name} (found in: {', '.join(splits_found)})")
            
    if low_train_classes:
        print("\nQUALITY WARNING: Classes with low (< 10) representation in train split:")
        for cls_name, count in low_train_classes.items():
            print(f"  - {cls_name}: {count} bounding boxes")
            
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze a YOLO dataset splits and label structures.")
    parser.add_argument("--dataset", type=str, default="datasets/plant_disease_v3", help="Path to the dataset root folder containing data.yaml")
    parser.add_argument("--output-report-dir", type=str, default="reports", help="Directory where JSON analysis report will be saved")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    report_dir = Path(args.output_report_dir)
    
    analysis = analyze_dataset(dataset_path)
    
    # Save JSON report
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"analysis_{dataset_path.name}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved detailed analysis report to: {report_path}")

if __name__ == "__main__":
    main()
