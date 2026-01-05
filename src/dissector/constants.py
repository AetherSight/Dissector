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

# 身体部位提示词 - Ultralytics 后端（完整版）
BODY_PARTS_PROMPTS_ULTRA = {
    "upper": [
        "upper body clothing",
        "upper garment",
        "top",
        "shirt",
        "blouse",
        "jacket",
        "coat",
        "sweater",
        "cardigan",
        "hoodie",
        "tunic",
        "vest",
        "armor chest",
        "breastplate",
        "dress bodice",
        "dress top",
        "upper part of dress",
        "sleeve",
        "long sleeve",
        "short sleeve",
        "arm guard",
        "bracer",
        "arm band",
        "arm accessory",
        "garment body",
        "clothing fabric",
        "inner lining",
    ],
    "lower_negation_for_upper": [
        "leg",
        "pants",
    ],
    "lower": [
        "pants",
        "trousers",
        "jeans",
        "slacks",
        "shorts",
        "leggings",
        "tights",
        "pant legs",
        "trouser legs",
        "pant waist",
    ],
    "shoes": [
        "shoe",
        "shoes",
        "boot",
        "boots",
        "sandal",
        "sneaker",
        "high heel",
        "flat shoe",
    ],
    "head": [
        "head",
        "human head",
        "face",
        "facial area",
        "hair",
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
    "hands": [
        "human hand",
        "hands",
        "palm",
        "fingers",
        "bare hand",
        "bare fingers",
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
