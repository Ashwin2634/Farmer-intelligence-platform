import os
import shutil
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict

def calculate_blurriness(image_path):
    """Computes the Laplacian variance to measure image sharpness."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0
        return cv2.Laplacian(image, cv2.CV_64F).var()
    except Exception:
        return 0

def get_image_hash(image_path, hash_size=8):
    """Generates a perceptual hash (dHash) to find exact or near-duplicates."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img.getdata(), dtype=float).reshape((hash_size, hash_size + 1))
            diff = pixels[:, 1:] > pixels[:, :-1]
            return "".join([str(int(b)) for b in diff.flatten()])
    except Exception:
        return None

def hamming_distance(hash1, hash2):
    """Measures how different two hashes are."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def extract_features(image_path):
    """Extracts a color histogram to use for general similarity matching."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception:
        return None

def reduce_image_folder(input_dir, output_dir, target_count=300):
    """Filters, deduplicates, and selects the top N highest-quality images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    total_found = len(all_files)
    print(f"📸 Found {total_found} images in the input directory.")
    
    if total_found <= target_count:
        print("⚠️ Input images are already fewer than or equal to the target count. Copying all.")
        for f in all_files:
            shutil.copy(f, output_dir)
        return

    # Step 1: Scan and calculate metrics (Sharpness & Hashes)
    print("⏳ Analyzing images for sharpness and duplicates...")
    image_data = []
    hash_buckets = defaultdict(list)
    
    for path in all_files:
        blur_score = calculate_blurriness(path)
        img_hash = get_image_hash(path)
        features = extract_features(path)
        
        if img_hash and features is not None:
            img_info = {
                'path': path,
                'blur': blur_score,
                'hash': img_hash,
                'features': features,
                'kept': True
            }
            image_data.append(img_info)
            hash_buckets[img_hash].append(img_info)

    # Step 2: Remove exact duplicates (Keep the sharpest one in each hash bucket)
    print("✂️ Removing exact duplicates...")
    for img_hash, copies in hash_buckets.items():
        if len(copies) > 1:
            copies.sort(key=lambda x: x['blur'], reverse=True)
            for duplicate in copies[1:]:
                duplicate['kept'] = False

    active_images = [img for img in image_data if img['kept']]
    print(f"📉 Reduced to {len(active_images)} images after exact duplicate removal.")

    # Step 3: Progressive similarity filtering to reach the target count
    # We sort by blurriness so we prioritize keeping sharper images over blurry ones
    active_images.sort(key=lambda x: x['blur'], reverse=True)
    
    # We dynamically find a similarity threshold if we are still over the target
    if len(active_images) > target_count:
        print("🤖 Clustered reduction: Removing highly similar images...")
        final_selection = []
        
        # We look at the list sequentially. Because it's sorted by blurriness, 
        # the best photos are checked first, and subsequent similar photos are dropped.
        for item in active_images:
            if len(final_selection) >= target_count:
                break
                
            is_similar = False
            for selected in final_selection:
                # Check color histogram similarity (1.0 is perfect match)
                sim_score = cv2.compareHist(
                    item['features'].reshape(8,8,8), 
                    selected['features'].reshape(8,8,8), 
                    cv2.HISTCMP_CORREL
                )
                
                # Check structural hash distance
                h_dist = hamming_distance(item['hash'], selected['hash'])
                
                # Strict thresholding: Adjust if you want tighter or looser groupings
                if sim_score > 0.85 or h_dist < 12:
                    is_similar = True
                    break
            
            if not is_similar:
                final_selection.append(item)
        
        # Step 4: Hard cutoff safety net
        # If the similarity filter was too strict and left too many images, 
        # take the top N absolute sharpest remaining images.
        if len(final_selection) > target_count:
            final_selection = final_selection[:target_count]
            
        # If similarity filter was too aggressive, backfill from the sharpest dropped ones
        if len(final_selection) < target_count:
            remaining_slots = target_count - len(final_selection)
            selected_paths = {x['path'] for x in final_selection}
            backfill_candidates = [x for x in active_images if x['path'] not in selected_paths]
            final_selection.extend(backfill_candidates[:remaining_slots])
    else:
        final_selection = active_images

    # Step 5: Export chosen files
    print(f"📦 Copying the final {len(final_selection)} selected images to '{output_dir}'...")
    for item in final_selection:
        shutil.copy(item['path'], output_dir)
        
    print("✅ Process complete!")

# ==========================================
# REUSABLE CONFIGURATION
# ==========================================
if __name__ == "__main__":
    # Get the directory where this script is running
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths relative to the script's root directory
    INPUT_FOLDER = os.path.join(SCRIPT_DIR, "plantseg_raw_new", "cucumber_bacterial_wilt") 
    OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "plantseg_raw_neww", "cucumber_bacterial_wilt")
    TARGET_IMAGE_COUNT = 100 

    reduce_image_folder(
        input_dir=INPUT_FOLDER, 
        output_dir=OUTPUT_FOLDER, 
        target_count=TARGET_IMAGE_COUNT
    )
