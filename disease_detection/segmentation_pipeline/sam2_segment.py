"""
sam2_segment.py

SAM2 inference module.
"""

import numpy as np

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import *
from preprocess import automatic_prompt_generation


# ==========================================================
# LOAD MODEL (only once)
# ==========================================================

_model = build_sam2_hf(
    SAM2_MODEL_NAME,
    device=DEVICE
)

_predictor = SAM2ImagePredictor(_model)


# ==========================================================
# SEGMENT ONE IMAGE
# ==========================================================

def segment_image(image):

    # Current preprocess.py returns:
    # mask, points

    _, points = automatic_prompt_generation(image)

    if len(points) == 0:
        return None, None, None

    labels = np.ones(len(points), dtype=np.int32)

    _predictor.set_image(image)

    masks, scores, logits = _predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )

    best = np.argmax(scores)

    final_mask = masks[best].astype(np.uint8)

    return final_mask, float(scores[best]), points

    # IMPORTANT:
    # Keep this line compatible with whichever preprocess.py
    # you're currently using.
    #
    # If your current automatic_prompt_generation() returns:
    #   candidate, cleaned, points
    # then use:
    #
    # candidate, cleaned, points = automatic_prompt_generation(image)
    #
    # If it returns:
    #   mask, points
    # then use:
    #
    # mask, points = automatic_prompt_generation(image)

    candidate, cleaned, points = automatic_prompt_generation(image)

    if len(points) == 0:
        return None, None, None

    labels = np.ones(len(points), dtype=np.int32)

    _predictor.set_image(image)

    masks, scores, logits = _predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )

    best = np.argmax(scores)

    final_mask = masks[best].astype(np.uint8)

    return final_mask, scores[best], points