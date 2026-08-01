"""
dataset.py

Loads the dataset and prepares train/val/test splits.
"""

from pathlib import Path
import random

from config import *

# ---------------------------------------------------
# CLASS DISCOVERY
# ---------------------------------------------------

def discover_classes():

    folders = [
        f.name
        for f in INPUT_DATASET.iterdir()
        if f.is_dir()
    ]

    folders = sorted(folders)

    class_to_id = {
        name: idx
        for idx, name in enumerate(folders)
    }

    id_to_class = {
        idx: name
        for name, idx in class_to_id.items()
    }

    return class_to_id, id_to_class


# ---------------------------------------------------
# IMAGE DISCOVERY
# ---------------------------------------------------

def load_dataset():

    class_to_id, id_to_class = discover_classes()

    dataset = []

    for class_name in sorted(class_to_id.keys()):

        folder = INPUT_DATASET / class_name

        for image_path in folder.rglob("*"):

            if image_path.suffix not in IMAGE_EXTENSIONS:
                continue

            dataset.append({

                "path": image_path,

                "filename": image_path.name,

                "stem": image_path.stem,

                "class_name": class_name,

                "class_id": class_to_id[class_name]

            })

    random.seed(RANDOM_SEED)

    random.shuffle(dataset)

    return dataset, class_to_id, id_to_class


# ---------------------------------------------------
# SPLIT
# ---------------------------------------------------

def split_dataset(dataset):

    total = len(dataset)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train = dataset[:train_end]

    val = dataset[train_end:val_end]

    test = dataset[val_end:]

    return train, val, test