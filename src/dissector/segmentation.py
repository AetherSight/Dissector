"""
身体部位分割模块
使用 SAM3 文本提示功能自动分割5个部位：upper, lower, shoes, head, hands
支持 MLX 和 Ultralytics 后端
"""
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, List, Tuple
import base64
import torch
import os

from .backend import SAM3Base

logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    char_count = len(text)
    tokens_by_words = int(word_count * 1.33) + 1
    tokens_by_chars = int(char_count / 4) + 1
    return max(tokens_by_words, tokens_by_chars)

BODY_PARTS_PROMPTS_CORE = {
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

BODY_PARTS_PROMPTS_FULL = {
    "upper": [
        "upper body clothing",
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
        "dress",
        "sleeve",
        "arm guard",
        "bracer",
        "arm band",
        "arm accessory",
        "scarf",
        "necklace",
        "collar",
        "bag",
        "handbag",
        "shoulder bag",
        "crossbody bag",
        "purse",
        "satchel",
        "messenger bag",
        "tote bag",
        "backpack",
        "rucksack",
        "pouch",
        "waist pouch",
        "hip pouch",
        "garment hem",
        "clothing hem",
        "bottom hem",
        "lower edge of clothing",
        "hem area",
        "attached accessory",
        "accessory attached to garment",
        "waist accessory",
        "hip accessory",
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
        "inner lining",
    ],
    "shoes": [
        "shoes",
        "boot",
        "boots",
        "sandal",
        "sneaker",
        "high heel",
        "flat shoe",
        "footwear",
    ],
    "head": [
        "head",
        "human head",
        "hair",
        "hairstyle",
        "face",
        "facial area",
        "facial features",
        "forehead",
        "cheeks",
        "chin",
        "jaw",
        "facial skin",
        "ponytail hair",
        "headwear",
        "headpiece",
        "headdress",
        "hat",
        "cap",
        "helmet",
        "crown",
        "tiara",
        "headband",
        "hood",
        "wig",
        "turban",
        "beret",
        "beanie",
        "bonnet",
        "visor",
        "baseball cap",
        "head wrap",
        "bandana",
        "veil",
        "head accessory",
        "hair accessory",
        "hair flower",
        "hair ornament",
        "hair clip",
        "hairpin",
        "hair band",
        "hair ribbon",
        "hair bow",
        "ears",
        "human ear",
        "earlobe",
        "ear accessory",
        "earring",
        "earrings",
        "ear jewelry",
        "cat ear",
        "animal ear",
    ],
    "hands": [
        "hands",
        "human hand",
        "palm",
        "fingers",
        "bare hand",
        "bare fingers",
        "wrist",
        "thumb",
        "hand accessory",
        "glove",
        "ring",
    ],
}

BODY_PARTS_PROMPTS = BODY_PARTS_PROMPTS_CORE

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

def segment_parts(
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
    
    debug_upper_dir = os.path.join("tmp", "debug_upper")
    debug_head_dir = os.path.join("tmp", "debug_head")
    os.makedirs(debug_upper_dir, exist_ok=True)
    os.makedirs(debug_head_dir, exist_ok=True)
    
    results: Dict[str, str] = {}
    
    use_dino = sam3_model.backend_name == "ultralytics" and processor is not None and dino_model is not None
    
    if sam3_model.backend_name == "mlx":
        prompts_dict = BODY_PARTS_PROMPTS_CORE
    else:
        prompts_dict = BODY_PARTS_PROMPTS_FULL
    
    masks_dict: Dict[str, np.ndarray] = {}
    
    other_parts = ["lower", "shoes", "head", "hands"]
    
    for part_name, prompts in prompts_dict.items():
        if part_name == "lower_negation_for_upper":
            continue
        try:
            logger.info(f"Segmenting {part_name}...")
            
            mask = None
            
            if sam3_model.backend_name == "mlx":
                logger.info(f"MLX: {part_name} processing {len(prompts)} individual prompts")
                
                mask_total = None
                successful_prompts = 0
                failed_prompts = 0
                
                for i, prompt in enumerate(prompts):
                    try:
                        prompt_mask = sam3_model.generate_mask_from_text_prompt(
                            image_pil=image_pil,
                            text_prompt=prompt,
                        )
                        if prompt_mask is not None and prompt_mask.size > 0:
                            mask_pixels = np.sum(prompt_mask)
                            if mask_pixels > 0:
                                if part_name == "upper":
                                    prompt_mask_vis = (prompt_mask.astype(np.uint8) * 255)
                                    prompt_overlay = image_bgr.copy()
                                    prompt_overlay[prompt_mask] = prompt_overlay[prompt_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
                                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
                                    cv2.imwrite(
                                        os.path.join(debug_upper_dir, f"upper_prompt_{i+1:02d}_{safe_prompt}_mask.png"),
                                        prompt_mask_vis
                                    )
                                    cv2.imwrite(
                                        os.path.join(debug_upper_dir, f"upper_prompt_{i+1:02d}_{safe_prompt}_overlay.jpg"),
                                        prompt_overlay
                                    )
                                elif part_name == "head":
                                    prompt_mask_vis = (prompt_mask.astype(np.uint8) * 255)
                                    prompt_overlay = image_bgr.copy()
                                    prompt_overlay[prompt_mask] = prompt_overlay[prompt_mask] * 0.5 + np.array([255, 0, 255], dtype=np.uint8) * 0.5
                                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
                                    cv2.imwrite(
                                        os.path.join(debug_head_dir, f"head_prompt_{i+1:02d}_{safe_prompt}_mask.png"),
                                        prompt_mask_vis
                                    )
                                    cv2.imwrite(
                                        os.path.join(debug_head_dir, f"head_prompt_{i+1:02d}_{safe_prompt}_overlay.jpg"),
                                        prompt_overlay
                                    )
                                
                                if mask_total is None:
                                    mask_total = prompt_mask.copy()
                                else:
                                    mask_total |= prompt_mask
                                successful_prompts += 1
                                logger.debug(f"MLX: {part_name} prompt {i+1}/{len(prompts)} '{prompt[:30]}...' succeeded, mask pixels: {mask_pixels}")
                            else:
                                failed_prompts += 1
                                logger.debug(f"MLX: {part_name} prompt {i+1}/{len(prompts)} '{prompt[:30]}...' returned empty mask")
                        else:
                            failed_prompts += 1
                            logger.debug(f"MLX: {part_name} prompt {i+1}/{len(prompts)} '{prompt[:30]}...' returned None or empty")
                    except Exception as e:
                        failed_prompts += 1
                        logger.warning(f"MLX: {part_name} prompt {i+1}/{len(prompts)} '{prompt[:30]}...' error: {e}")
                        continue
                
                if mask_total is not None and mask_total.size > 0:
                    total_pixels = np.sum(mask_total)
                    logger.info(f"MLX: {part_name} completed: {successful_prompts} successful, {failed_prompts} failed prompts, total mask pixels: {total_pixels}")
                else:
                    logger.warning(f"MLX: {part_name} failed: no valid mask generated from {len(prompts)} prompts")
                
                mask = mask_total
                
                if (mask is None or mask.size == 0) and use_dino:
                    logger.info(f"MLX text prompt failed for {part_name}, falling back to DINO + SAM3")
                    boxes = dino_detect(
                        image_pil=image_pil,
                        prompts=prompts,
                        processor=processor,
                        dino_model=dino_model,
                        device=device,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                    )
                    
                    if boxes.size > 0:
                        mask = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
                    else:
                        logger.warning(f"No boxes detected for {part_name}")
            else:
                if use_dino:
                    logger.info(f"Using DINO + SAM3 for {part_name} (Ultralytics backend)")
                    boxes = dino_detect(
                        image_pil=image_pil,
                        prompts=prompts,
                        processor=processor,
                        dino_model=dino_model,
                        device=device,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                    )
                    
                    if boxes.size > 0:
                        mask = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
                    else:
                        logger.warning(f"No boxes detected for {part_name}")
                else:
                    logger.warning(f"DINO not available for {part_name} (Ultralytics backend)")
                    mask = None
            
            if mask is None or mask.size == 0 or np.sum(mask) == 0:
                logger.warning(f"No valid mask found for {part_name} (mask is None or empty)")
                masks_dict[part_name] = np.zeros((h, w), dtype=bool)
                continue
            
            if mask.shape != (h, w):
                mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
                mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            masks_dict[part_name] = mask
            
        except Exception as e:
            logger.error(f"Error segmenting {part_name}: {e}", exc_info=True)
            masks_dict[part_name] = np.zeros((h, w), dtype=bool)
    
    if "lower_negation_for_upper" in prompts_dict:
        logger.info("Detecting lower_negation_for_upper for upper mask calculation...")
        lower_negation_prompts = prompts_dict["lower_negation_for_upper"]
        lower_negation_mask = None
        
        if sam3_model.backend_name == "mlx":
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
                    logger.warning(f"Error detecting lower_negation_for_upper with prompt '{prompt}': {e}")
                    continue
            lower_negation_mask = lower_negation_mask_total
        else:
            if use_dino:
                boxes = dino_detect(
                    image_pil=image_pil,
                    prompts=lower_negation_prompts,
                    processor=processor,
                    dino_model=dino_model,
                    device=device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                if boxes.size > 0:
                    lower_negation_mask = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
        
        if lower_negation_mask is None or lower_negation_mask.size == 0:
            lower_mask = masks_dict.get("lower", np.zeros((h, w), dtype=bool))
            lower_negation_mask = lower_mask.copy()
        else:
            if lower_negation_mask.shape != (h, w):
                mask_uint8 = (lower_negation_mask.astype(np.uint8) * 255) if lower_negation_mask.dtype == bool else lower_negation_mask.astype(np.uint8)
                lower_negation_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        masks_dict["lower_negation_for_upper"] = lower_negation_mask
    else:
        masks_dict["lower_negation_for_upper"] = masks_dict.get("lower", np.zeros((h, w), dtype=bool))
    
    logger.info("Detecting person/human body as a whole...")
    person_prompts = ["person", "full body", "outfit", "garment"]
    person_mask = None
    
    if sam3_model.backend_name == "mlx":
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
                logger.warning(f"Error detecting person with prompt '{prompt}': {e}")
                continue
        person_mask = person_mask_total
    else:
        if use_dino:
            boxes = dino_detect(
                image_pil=image_pil,
                prompts=person_prompts,
                processor=processor,
                dino_model=dino_model,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if boxes.size > 0:
                person_mask = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
    
    if person_mask is None or person_mask.size == 0 or np.sum(person_mask) == 0:
        logger.warning("Failed to detect person body, using fallback: all detected parts combined")
        person_mask = np.zeros((h, w), dtype=bool)
        for part_name in other_parts:
            if part_name in masks_dict:
                person_mask |= masks_dict[part_name]
    else:
        if person_mask.shape != (h, w):
            mask_uint8 = (person_mask.astype(np.uint8) * 255) if person_mask.dtype == bool else person_mask.astype(np.uint8)
            person_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    logger.info("Computing upper mask: cleaning detected upper, then merging with person body minus other parts...")
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
    
    upper_detected_vis = (upper_detected.astype(np.uint8) * 255)
    upper_detected_overlay = image_bgr.copy()
    upper_detected_overlay[upper_detected] = upper_detected_overlay[upper_detected] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_detected_mask.png"), upper_detected_vis)
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_detected_overlay.jpg"), upper_detected_overlay)
    
    upper_detected_cleaned_vis = (upper_detected_cleaned.astype(np.uint8) * 255)
    upper_detected_cleaned_overlay = image_bgr.copy()
    upper_detected_cleaned_overlay[upper_detected_cleaned] = upper_detected_cleaned_overlay[upper_detected_cleaned] * 0.5 + np.array([0, 255, 255], dtype=np.uint8) * 0.5
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_detected_cleaned_mask.png"), upper_detected_cleaned_vis)
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_detected_cleaned_overlay.jpg"), upper_detected_cleaned_overlay)
    
    upper_subtraction_vis = (upper_from_subtraction.astype(np.uint8) * 255)
    upper_subtraction_overlay = image_bgr.copy()
    upper_subtraction_overlay[upper_from_subtraction] = upper_subtraction_overlay[upper_from_subtraction] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_subtraction_mask.png"), upper_subtraction_vis)
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_subtraction_overlay.jpg"), upper_subtraction_overlay)
    
    upper_mask_vis = (upper_mask.astype(np.uint8) * 255)
    upper_overlay = image_bgr.copy()
    upper_overlay[upper_mask] = upper_overlay[upper_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_final_mask.png"), upper_mask_vis)
    cv2.imwrite(os.path.join(debug_upper_dir, "upper_final_overlay.jpg"), upper_overlay)
    
    person_mask_vis = (person_mask.astype(np.uint8) * 255)
    person_overlay = image_bgr.copy()
    person_overlay[person_mask] = person_overlay[person_mask] * 0.5 + np.array([255, 255, 0], dtype=np.uint8) * 0.5
    cv2.imwrite(os.path.join(debug_upper_dir, "person_body_mask.png"), person_mask_vis)
    cv2.imwrite(os.path.join(debug_upper_dir, "person_body_overlay.jpg"), person_overlay)
    
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
        logger.info(f"Successfully segmented {part_name}")
    
    return results
