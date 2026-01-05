"""
身体部位分割模块
使用 SAM3 文本提示功能自动分割5个部位：upper, lower, shoes, head, hands
支持 MLX 和 Ultralytics 后端
"""
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, List, Tuple
import base64
import torch

from .backend import SAM3Base

def estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    char_count = len(text)
    tokens_by_words = int(word_count * 1.33) + 1
    tokens_by_chars = int(char_count / 4) + 1
    return max(tokens_by_words, tokens_by_chars)

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

BODY_PARTS_PROMPTS = BODY_PARTS_PROMPTS_MIX

def white_bg(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.full_like(image_bgr, 255, dtype=np.uint8)
    
    mask_uint8 = (mask.astype(np.uint8)) * 255
    ys, xs = np.where(mask_uint8 > 0)
    
    if len(ys) == 0:
        return np.full_like(image_bgr, 255, dtype=np.uint8)
    
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = 5
    y0 = max(0, y0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    
    img_crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    mask_crop = mask_uint8[y0 : y1 + 1, x0 : x1 + 1]
    
    fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
    white = np.full_like(img_crop, 255, dtype=np.uint8)
    bg = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask_crop))
    return cv2.add(fg, bg)

def encode_image(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_uint8 = mask.astype(np.uint8)
    else:
        mask_uint8 = mask
    h, w = mask.shape
    min_area = max(1, int(h * w * min_area_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask
    keep = np.zeros(num_labels, dtype=bool)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[i] = True
    cleaned = keep[labels]
    return cleaned.astype(bool)

def dino_detect(
    image_pil: Image.Image,
    prompts: List[str],
    processor,
    dino_model,
    device: torch.device,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> np.ndarray:
    text = ". ".join(prompts) + "."
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    width, height = image_pil.size
    target_sizes = torch.tensor([[height, width]], device=device)
    
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=box_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = processor.post_process_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_sizes,
            )[0]
    
    boxes = results["boxes"].cpu().numpy() if "boxes" in results else np.array([])
    return boxes

def segment_parts_mlx(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    processor=None,
    dino_model=None,
    device: Optional[torch.device] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Dict[str, str]:
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]
    
    results: Dict[str, str] = {}
    prompts_dict = BODY_PARTS_PROMPTS_MIX
    masks_dict: Dict[str, np.ndarray] = {}
    other_parts = ["lower", "shoes", "head", "hands"]
    
    for part_name, prompts in prompts_dict.items():
        if part_name == "lower_negation_for_upper":
            continue
        try:
            mask_total = None
            for prompt in prompts:
                try:
                    prompt_mask = sam3_model.generate_mask_from_text_prompt(
                        image_pil=image_pil,
                        text_prompt=prompt,
                    )
                    if prompt_mask is not None and prompt_mask.size > 0:
                        mask_pixels = np.sum(prompt_mask)
                        if mask_pixels > 0:
                            if mask_total is None:
                                mask_total = prompt_mask.copy()
                            else:
                                mask_total |= prompt_mask
                except Exception as e:
                    continue
            
            mask = mask_total
            
            if mask is None or mask.size == 0 or np.sum(mask) == 0:
                masks_dict[part_name] = np.zeros((h, w), dtype=bool)
                continue
            
            if mask.shape != (h, w):
                mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
                mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            masks_dict[part_name] = mask
            
        except Exception as e:
            masks_dict[part_name] = np.zeros((h, w), dtype=bool)
    
    if "lower_negation_for_upper" in prompts_dict:
        lower_negation_prompts = prompts_dict["lower_negation_for_upper"]
        lower_negation_mask_total = None
        for prompt in lower_negation_prompts:
            try:
                prompt_mask = sam3_model.generate_mask_from_text_prompt(
                    image_pil=image_pil,
                    text_prompt=prompt,
                )
                if prompt_mask is not None and prompt_mask.size > 0 and np.sum(prompt_mask) > 0:
                    if lower_negation_mask_total is None:
                        lower_negation_mask_total = prompt_mask.copy()
                    else:
                        lower_negation_mask_total |= prompt_mask
            except Exception as e:
                continue
        
        if lower_negation_mask_total is None or lower_negation_mask_total.size == 0:
            lower_negation_mask = masks_dict.get("lower", np.zeros((h, w), dtype=bool)).copy()
        else:
            lower_negation_mask = lower_negation_mask_total
            if lower_negation_mask.shape != (h, w):
                mask_uint8 = (lower_negation_mask.astype(np.uint8) * 255) if lower_negation_mask.dtype == bool else lower_negation_mask.astype(np.uint8)
                lower_negation_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        masks_dict["lower_negation_for_upper"] = lower_negation_mask
    else:
        masks_dict["lower_negation_for_upper"] = masks_dict.get("lower", np.zeros((h, w), dtype=bool))
    
    person_prompts = ["person", "full body", "outfit", "garment"]
    person_mask_total = None
    for prompt in person_prompts:
        try:
            prompt_mask = sam3_model.generate_mask_from_text_prompt(
                image_pil=image_pil,
                text_prompt=prompt,
            )
            if prompt_mask is not None and prompt_mask.size > 0 and np.sum(prompt_mask) > 0:
                if person_mask_total is None:
                    person_mask_total = prompt_mask.copy()
                else:
                    person_mask_total |= prompt_mask
        except Exception as e:
            continue
    
    if person_mask_total is None or person_mask_total.size == 0:
        person_mask = np.zeros((h, w), dtype=bool)
        for part_name in other_parts:
            if part_name in masks_dict:
                person_mask |= masks_dict[part_name]
    else:
        person_mask = person_mask_total
        if person_mask.shape != (h, w):
            mask_uint8 = (person_mask.astype(np.uint8) * 255) if person_mask.dtype == bool else person_mask.astype(np.uint8)
            person_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    upper_detected = masks_dict.get("upper", np.zeros((h, w), dtype=bool))
    lower_negation_mask = masks_dict.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))
    
    upper_detected_cleaned = upper_detected.copy()
    upper_detected_cleaned = upper_detected_cleaned & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_detected_cleaned = upper_detected_cleaned & (~masks_dict[part_name])
    
    upper_from_subtraction = person_mask.copy()
    upper_from_subtraction = upper_from_subtraction & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_from_subtraction = upper_from_subtraction & (~masks_dict[part_name])
    
    upper_mask = upper_detected_cleaned | upper_from_subtraction
    upper_mask = clean_mask(upper_mask, min_area_ratio=0.001)
    masks_dict["upper"] = upper_mask
    
    if "lower" in masks_dict:
        lower_mask = masks_dict["lower"].copy()
        if "shoes" in masks_dict:
            lower_mask = lower_mask & (~masks_dict["shoes"])
        lower_mask = clean_mask(lower_mask, min_area_ratio=0.001)
        masks_dict["lower"] = lower_mask
    
    for part_name in prompts_dict.keys():
        mask = masks_dict.get(part_name, np.zeros((h, w), dtype=bool))
        if part_name not in ["upper", "lower"]:
            mask = clean_mask(mask, min_area_ratio=0.001)
        cropped_img = white_bg(image_bgr, mask)
        results[part_name] = encode_image(cropped_img, ext=".jpg")
    
    return results

def segment_parts(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    processor=None,
    dino_model=None,
    device: Optional[torch.device] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Dict[str, str]:
    # segment_parts 只用于 MLX 后端，ultralytics 后端使用 process_image_ultralytics
    return segment_parts_mlx(
        image_pil=image_pil,
        sam3_model=sam3_model,
        processor=processor,
        dino_model=dino_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
