"""
config.py

Central configuration for the automatic SAM2 -> YOLO Segmentation pipeline.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/AI_Service")

PIPELINE_ROOT = PROJECT_ROOT / "segmentation_pipeline"

INPUT_DATASET = PROJECT_ROOT / "datasets" / "plant_seg_new"

YOLO_DATASET = PROJECT_ROOT / "datasets" / "plant_seg_yolo"

OUTPUT_ROOT = PIPELINE_ROOT / "output"

MASK_DIR = OUTPUT_ROOT / "masks"

DEBUG_DIR = OUTPUT_ROOT / "debug"

LOG_DIR = PIPELINE_ROOT / "logs"

IMAGE_OUTPUT = YOLO_DATASET / "images"

LABEL_OUTPUT = YOLO_DATASET / "labels"

# =============================================================================
# OUTPUT DATASET
# =============================================================================

TRAIN_IMAGES = IMAGE_OUTPUT / "train"
VAL_IMAGES = IMAGE_OUTPUT / "val"
TEST_IMAGES = IMAGE_OUTPUT / "test"

TRAIN_LABELS = LABEL_OUTPUT / "train"
VAL_LABELS = LABEL_OUTPUT / "val"
TEST_LABELS = LABEL_OUTPUT / "test"

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

DIRECTORIES = [

    OUTPUT_ROOT,

    MASK_DIR,
    DEBUG_DIR,
    LOG_DIR,

    IMAGE_OUTPUT,
    LABEL_OUTPUT,

    TRAIN_IMAGES,
    VAL_IMAGES,
    TEST_IMAGES,

    TRAIN_LABELS,
    VAL_LABELS,
    TEST_LABELS,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# IMAGE SETTINGS
# =============================================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",

    ".JPG",
    ".JPEG",
    ".PNG",

    ".bmp",
    ".BMP",

    ".webp",
    ".WEBP",
}

# =============================================================================
# SAM2 SETTINGS
# =============================================================================

SAM2_MODEL_NAME = "facebook/sam2-hiera-tiny"

DEVICE = "cuda"

MULTI_MASK = True

TOP_K_COMPONENTS = 3

# =============================================================================
# PREPROCESSING
# =============================================================================

SATURATION_THRESHOLD = 35

VALUE_THRESHOLD = 210

LAB_A_THRESHOLD = 135

MORPH_KERNEL = 5

MIN_COMPONENT_AREA = 500

# =============================================================================
# POLYGON
# =============================================================================

POLYGON_EPSILON = 0.003

MIN_POLYGON_POINTS = 6

# =============================================================================
# DATASET SPLIT
# =============================================================================

TRAIN_RATIO = 0.80

VAL_RATIO = 0.10

TEST_RATIO = 0.10

RANDOM_SEED = 42

# =============================================================================
# DEBUG
# =============================================================================

SAVE_MASKS = True

SAVE_DEBUG_IMAGES = True

PRINT_PROGRESS = True

# =============================================================================

print("Configuration Loaded")