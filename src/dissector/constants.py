"""
常量定义
包含提示词、阈值等配置
"""
from typing import Dict, List

# 身体部位提示词 - MLX 后端（精简版）
BODY_PARTS_PROMPTS_MIX = {
    "upper": [
        "upper body clothing",
        "waist belt",
        "fabric",
        "accessory",
        "dress",
        "skirt",
    ],
    "lower_negation_for_upper": [
        "leg",
        "pants",
    ],
    "lower": [
        "leg",
        "pants",
        "skirt",
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
    ],
    "hands": [
        "hands",
        "gloves",
        "ring",
    ],
}

BODY_PARTS_PROMPTS_ULTRA = {
    **BODY_PARTS_PROMPTS_MIX,
    "head": [
        "head",
        "hair",
        "face",
        "head hair accessory",
        "ear",
        "earring",
        "human head",
        "facial area",
        "hair accessory",
        "hairstyle",
        "ponytail hair",
        "cat ear",
        "animal ear",
        "headwear",
        "hat",
        "cap",
        "helmet",
        "crown",
        "tiara",
        "headband",
        "hood",
    ],
}

# 默认阈值配置
DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.25

# Mask 处理配置
DEFAULT_MIN_AREA_RATIO = 0.001
HANDS_MIN_AREA_RATIO = 0.0005

# 形态学操作配置
HEAD_DILATE_KERNEL_SIZE = 15
HANDS_DILATE_KERNEL_SIZE = 5
MASK_CLOSE_KERNEL_SIZE = 3
