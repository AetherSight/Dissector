"""
Constants definition
Contains prompts, thresholds and other configurations
"""
from typing import Dict, List

# Body parts prompts - MLX backend (simplified version)
BODY_PARTS_PROMPTS_MIX = {
    "upper": [
        "upper body clothing",
        "waist belt",
        "fabric",
        "accessory",
        "dress",
        "skirt",
        "clothing strap"
    ],
    "lower_negation_for_upper": [
        "leg",
        "pants",
        "socks",
    ],
    "lower": [
        "leg",
        "pants",
        "skirt",
        "socks",
    ],
    "shoes": [
        "footwear",
        "shoes",
    ],
    "head": [
        "head",
        "hair",
        "face",
        "head hair accessory",
        "ear",
        "earring",
        "hat",
        "hood",
    ],
    "hands": [
        "hands",
        "gloves",
        "ring",
    ],
}

# Body parts prompts - Ultralytics backend (full version, kept as alternative)
BODY_PARTS_PROMPTS_ULTRA = {
    **BODY_PARTS_PROMPTS_MIX,
    "head": [
        ["headwear", "head accessory", "face"],
        ["hair", "hairband", "ears", "earring", "triangular ears"],
    ],
}

# Default threshold configuration
DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.25

# Mask processing configuration
DEFAULT_MIN_AREA_RATIO = 0.001
HANDS_MIN_AREA_RATIO = 0.0005

# Morphological operation configuration
HEAD_DILATE_KERNEL_SIZE = 15
HANDS_DILATE_KERNEL_SIZE = 5
MASK_CLOSE_KERNEL_SIZE = 3
