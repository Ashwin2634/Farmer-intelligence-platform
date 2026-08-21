#!/usr/bin/env python3
import os
import sys
import csv
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    dataset_dir = Path("datasets/plantseg_raw")
    metadata_csv_path = dataset_dir / "Metadata.csv"
    
    if not metadata_csv_path.exists():
        print(f"Error: {metadata_csv_path} not found.")
        sys.exit(1)
        
    output_dir = Path("reports/plantseg_analysis/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We will pick a few representative crops and diseases for visualization
    targets = [
        {"crop": "Apple", "disease": "apple black rot"},
        {"crop": "Apple", "disease": "apple rust"},
        {"crop": "Tomato", "disease": "tomato early blight"},
        {"crop": "Tomato", "disease": "tomato late blight"},
        {"crop": "Bell pepper", "disease": "bell pepper bacterial spot"},
        {"crop": "Corn", "disease": "corn rust"},
        {"crop": "Grape", "disease": "grape downy mildew"},
        {"crop": "Potato", "disease": "potato early blight"}
    ]
    
    # Read metadata to find matching images
    selected_samples = []
    
    # Load all records
    records = []
    try:
        with open(metadata_csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"Error reading Metadata.csv: {e}")
        sys.exit(1)
        
    # Group images by split and name on disk
    disk_images = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / "images" / split
        if split_dir.exists():
            for p in split_dir.glob("*"):
                if p.is_file():
                    disk_images[p.name] = (split, p)

    disk_masks = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / "annotations" / split
        if split_dir.exists():
            for p in split_dir.glob("*"):
                if p.is_file():
                    disk_masks[p.name] = (split, p)

    # For each target crop/disease, pick one valid sample
    for target in targets:
        target_crop = target["crop"]
        target_disease = target["disease"]
        
        for row in records:
            if row["Plant"] == target_crop and row["Disease"] == target_disease:
                img_name = row["Name"]
                mask_name = row["Label file"]
                
                if img_name in disk_images and mask_name in disk_masks:
                    selected_samples.append({
                        "crop": target_crop,
                        "disease": target_disease,
                        "img_name": img_name,
                        "mask_name": mask_name,
                        "img_path": disk_images[img_name][1],
                        "mask_path": disk_masks[mask_name][1]
                    })
                    break # just pick the first match

    print(f"Selected {len(selected_samples)} samples for visualization.")
    
    for idx, sample in enumerate(selected_samples):
        crop = sample["crop"]
        disease = sample["disease"]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]
        
        print(f"Visualizing: {crop} - {disease} ({img_path.name})")
        
        try:
            # Open image and convert to RGB
            img = Image.open(img_path).convert("RGB")
            # Open mask
            msk = Image.open(mask_path).convert("L")
            
            # Verify size matches
            if img.size != msk.size:
                print(f"  Warning: size mismatch for {img_path.name}, resizing mask to match image.")
                msk = msk.resize(img.size, Image.Resampling.NEAREST)
                
            img_arr = np.array(img)
            msk_arr = np.array(msk)
            
            # Create a red overlay color representation
            overlay_arr = img_arr.copy()
            
            # The mask values are binary 0 and 1
            mask_pixels = msk_arr > 0
            
            # Overlay translucent red (alpha blend: 60% original image, 40% red)
            overlay_arr[mask_pixels, 0] = (img_arr[mask_pixels, 0] * 0.6 + 255 * 0.4).astype(np.uint8)
            overlay_arr[mask_pixels, 1] = (img_arr[mask_pixels, 1] * 0.6 + 0 * 0.4).astype(np.uint8)
            overlay_arr[mask_pixels, 2] = (img_arr[mask_pixels, 2] * 0.6 + 0 * 0.4).astype(np.uint8)
            
            # Concatenate horizontally (Original | Mask | Overlay)
            # Create a visual representation of mask as grey scale 0-255
            msk_visual_arr = np.stack([msk_arr * 255] * 3, axis=-1)
            
            combined_h = np.hstack([img_arr, msk_visual_arr, overlay_arr])
            combined_img = Image.fromarray(combined_h)
            
            # Save the result
            clean_disease_name = disease.replace(" ", "_").replace("(", "").replace(")", "")
            output_name = f"{crop.lower()}_{clean_disease_name}_visual.jpg"
            combined_img.save(output_dir / output_name, quality=90)
            print(f"  Saved visual QA to {output_dir / output_name}")
            
        except Exception as e:
            print(f"  Error visualising {img_path.name}: {e}")

if __name__ == "__main__":
    main()
