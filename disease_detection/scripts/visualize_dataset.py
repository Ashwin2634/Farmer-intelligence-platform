import os
import sys
import yaml
import random
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from PIL import Image, ImageDraw, ImageColor, ImageFont

def parse_simple_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Parses data.yaml."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_class_colors(nc: int) -> List[Tuple[int, int, int]]:
    """Generates distinct RGB colors for each class using HSL color space."""
    colors = []
    for i in range(nc):
        hue = int(i * 360 / max(nc, 1))
        # Convert HSL to RGB dynamically using PIL's ImageColor
        color_rgb = ImageColor.getrgb(f"hsl({hue}, 90%, 50%)")
        colors.append(color_rgb)
    return colors

def draw_annotations(img_path: Path, label_path: Path, class_names: List[str], colors: List[Tuple[int, int, int]]) -> Image.Image:
    """Draws YOLO bounding boxes and labels onto the image."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    draw = ImageDraw.Draw(img)
    
    # Try to load a default font, otherwise fall back to PIL basic font
    try:
        # standard system fonts
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    if not label_path.exists():
        return img
        
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
            
        cls_id = int(parts[0])
        x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])
        
        # Denormalize coordinates
        x1 = int((x_center - bbox_w / 2.0) * w)
        y1 = int((y_center - bbox_h / 2.0) * h)
        x2 = int((x_center + bbox_w / 2.0) * w)
        y2 = int((y_center + bbox_h / 2.0) * h)
        
        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        # Get color
        color = colors[cls_id] if cls_id < len(colors) else (255, 255, 255)
        
        # Draw box
        # Calculate thickness based on image size (min 2px)
        thickness = max(2, int(min(w, h) / 150))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Draw label banner
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
        label_text = f"{class_name} (ID:{cls_id})"
        
        # Get text size
        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)
            
        # Draw small filled background box for text legibility
        text_bg_y1 = max(0, y1 - text_h - 4)
        text_bg_y2 = y1
        text_bg_x2 = min(w - 1, x1 + text_w + 6)
        
        draw.rectangle([x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
        
        # Write text in contrast color (white or black depending on brightness)
        # Simple brightness formula
        r, g, b = color
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        draw.text((x1 + 3, text_bg_y1 + 1), label_text, fill=text_color, font=font)
        
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset annotations.")
    parser.add_argument("--dataset", type=str, default="datasets/plant_disease_clean", help="Dataset directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Dataset split to visualize")
    parser.add_argument("--samples", type=str, default="10", help="Number of random samples to visualize")
    parser.add_argument("--balanced", action="store_true", help="Class-balanced visualization mode")
    parser.add_argument("--samples-per-class", type=int, default=10, help="Target samples per class in balanced mode")
    parser.add_argument("--class-name", type=str, default=None, help="Filter for target class name")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset).resolve()
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Error: data.yaml not found at {yaml_path}")
        sys.exit(1)
        
    yaml_data = parse_simple_yaml(yaml_path)
    class_names = yaml_data.get("names", [])
    nc = yaml_data.get("nc", len(class_names))
    colors = get_class_colors(nc)
    
    # Resolve split paths relative to data.yaml
    splits = {}
    for k in ["train", "val", "valid", "test"]:
        if k in yaml_data:
            path_val = yaml_data[k]
            # Resolve relative to dataset directory or yaml parent
            resolved_path = (yaml_path.parent / path_val).resolve()
            if resolved_path.exists():
                splits[k] = resolved_path
            else:
                # Fallback to direct name under dataset root
                direct_path = dataset_dir / Path(path_val).name
                if direct_path.exists():
                    splits[k] = direct_path
                else:
                    # Check standard split folders
                    std_path = dataset_dir / ('valid' if k == 'val' else k)
                    if std_path.exists():
                        splits[k] = std_path
                        
    # Map 'valid' or 'val' split input
    target_splits = [args.split]
    if args.split == "valid" and "val" in splits and "valid" not in splits:
        splits["valid"] = splits["val"]
        
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    manifest = []
    visualized_count = 0
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Class mapping lookup for class names
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    if args.class_name and args.class_name not in class_to_id:
        print(f"Error: Class '{args.class_name}' not found in data.yaml classes: {class_names}")
        sys.exit(1)
        
    # Helper to parse classes in labels
    def get_classes_in_label(label_file: Path) -> Set[int]:
        classes = set()
        if label_file.exists():
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            classes.add(int(line.split()[0]))
            except Exception:
                pass
        return classes

    # Helper to count boxes in labels
    def count_boxes_in_label(label_file: Path) -> int:
        if not label_file.exists():
            return 0
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                return len([line for line in f if line.strip()])
        except Exception:
            return 0

    if args.balanced:
        print("\nStarting Class-Balanced visual sampling...")
        # Save output in reports/visualizations/by_class/<class_name>/
        out_root = reports_dir / "visualizations" / "by_class"
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        
        # We need to scan all splits for images containing each class
        all_samples_by_class = {cid: [] for cid in range(nc)}
        
        for sp_name, sp_path in splits.items():
            # sp_path is image path or parent directory
            img_dir = sp_path
            lbl_dir = sp_path.parent / "labels"
            if img_dir.name == "images":
                lbl_dir = img_dir.parent / "labels"
            elif (img_dir / "images").exists():
                lbl_dir = img_dir / "labels"
                img_dir = img_dir / "images"
                
            if not img_dir.exists():
                continue
                
            for img_file in img_dir.iterdir():
                if img_file.suffix in img_extensions:
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    classes = get_classes_in_label(lbl_file)
                    for cid in classes:
                        if cid in all_samples_by_class:
                            all_samples_by_class[cid].append((img_file, lbl_file, sp_name))
                            
        # Now sample for each class
        underrepresented_classes = []
        
        for cid in range(nc):
            class_name = class_names[cid]
            candidates = all_samples_by_class[cid]
            total_available = len(candidates)
            
            print(f"Class '{class_name}' (ID {cid}): {total_available} candidate images available.")
            if total_available < args.samples_per_class:
                underrepresented_classes.append(class_name)
                
            # Randomly sample
            sampled = random.sample(candidates, min(total_available, args.samples_per_class))
            
            class_out_dir = out_root / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (img_file, lbl_file, sp) in enumerate(sampled, 1):
                annotated_img = draw_annotations(img_file, lbl_file, class_names, colors)
                
                # Output filename (shorten to avoid Windows MAX_PATH limitations)
                stem_limit = 50
                stem_part = img_file.stem
                if len(stem_part) > stem_limit:
                    import hashlib
                    h = hashlib.md5(img_file.stem.encode()).hexdigest()[:8]
                    stem_part = f"{img_file.stem[:stem_limit]}_{h}"
                out_filename = f"{stem_part}_class_{class_name}_{idx}.jpg"
                out_path = class_out_dir / out_filename
                
                annotated_img.save(out_path)
                visualized_count += 1
                
                # Manifest entry
                classes_present = [class_names[c] for c in get_classes_in_label(lbl_file)]
                manifest.append({
                    "visualization_file": str(out_path.resolve().relative_to(Path.cwd().resolve())),
                    "original_image_path": str(img_file),
                    "split": sp,
                    "classes_present": classes_present,
                    "boxes_count": count_boxes_in_label(lbl_file)
                })
                
        print(f"\nSuccessfully generated {visualized_count} class-balanced sample visualizations.")
        print(f"Saved manifest to reports/visual_inspection_manifest.json")
        if underrepresented_classes:
            print(f"Warning: The following classes had fewer than {args.samples_per_class} representative images:")
            for u_cls in underrepresented_classes:
                print(f"  - {u_cls}")
                
    else:
        # Standard Split/Filter Mode
        split_name = args.split
        sp_path = splits.get(split_name)
        if not sp_path:
            print(f"Error: Split path '{split_name}' not defined in data.yaml splits: {list(splits.keys())}")
            sys.exit(1)
            
        img_dir = sp_path
        lbl_dir = sp_path.parent / "labels"
        if img_dir.name == "images":
            lbl_dir = img_dir.parent / "labels"
        elif (img_dir / "images").exists():
            lbl_dir = img_dir / "labels"
            img_dir = img_dir / "images"
            
        if not img_dir.exists():
            print(f"Error: Image directory {img_dir} does not exist.")
            sys.exit(1)
            
        # Find images
        all_images = [p for p in img_dir.iterdir() if p.suffix in img_extensions]
        
        # Filter by class name if specified
        if args.class_name:
            target_cid = class_to_id[args.class_name]
            filtered_images = []
            for img_file in all_images:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if target_cid in get_classes_in_label(lbl_file):
                    filtered_images.append(img_file)
            all_images = filtered_images
            print(f"Filtered to {len(all_images)} images containing class '{args.class_name}'.")
            
        # Parse samples argument (can be 'all' or integer)
        if args.samples.lower() == 'all':
            sample_size = len(all_images)
        else:
            try:
                sample_size = min(int(args.samples), len(all_images))
            except ValueError:
                sample_size = min(10, len(all_images))
                
        print(f"Sampling {sample_size} images out of {len(all_images)} available from split '{split_name}'.")
        
        sampled_imgs = random.sample(all_images, sample_size)
        
        out_dir = reports_dir / "visualizations" / split_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in sampled_imgs:
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            annotated_img = draw_annotations(img_file, lbl_file, class_names, colors)
            
            # Shorten filename to avoid Windows MAX_PATH limitations
            short_name = img_file.name
            if len(img_file.name) > 80:
                import hashlib
                h = hashlib.md5(img_file.stem.encode()).hexdigest()[:8]
                short_name = f"{img_file.stem[:60]}_{h}{img_file.suffix}"
            out_path = out_dir / short_name
            annotated_img.save(out_path)
            visualized_count += 1
            
            classes_present = [class_names[c] for c in get_classes_in_label(lbl_file)]
            manifest.append({
                "visualization_file": str(out_path.resolve().relative_to(Path.cwd().resolve())),
                "original_image_path": str(img_file),
                "split": split_name,
                "classes_present": classes_present,
                "boxes_count": count_boxes_in_label(lbl_file)
            })
            
        print(f"Successfully generated {visualized_count} visualizations under: {out_dir}")
        
    # Write manifest file
    manifest_path = reports_dir / "visual_inspection_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
if __name__ == "__main__":
    main()
