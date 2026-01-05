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
        "shirt",
        "jacket",
        "coat",
        "sweater",
        "sleeve",
        "scarf",
        "belt",
        "waistband",
        "waist belt",
        "belt buckle",
        "bag",
        "handbag",
        "backpack",
    ],
    "lower": [
        "pants",
        "trousers",
        "jeans",
        "shorts",
        "leggings",
    ],
    "shoes": [
        "shoe",
        "shoes",
        "boot",
        "boots",
        "sandal",
    ],
    "head": [
        "head",
        "human head",
        "hair",
        "hairstyle",
        "face",
        "facial features",
        "headwear",
        "hat",
        "cap",
        "helmet",
        "ear",
        "earring",
    ],
    "hands": [
        "hands",
        "human hand",
        "hand",
        "fingers",
        "bare hand",
    ],
}

BODY_PARTS_PROMPTS_FULL = {
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
        "garment body",
        "clothing fabric",
        "inner lining",
        "lining",
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
        "lining",
        "inner lining",
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
        "facial region",
        "facial part",
        "forehead",
        "cheek",
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
        "head hair accessory",
        "head accessory",
        "hair accessory",
        "hair flower",
        "flower hair accessory",
        "hair ornament",
        "hair clip",
        "hairpin",
        "hair band",
        "hair ribbon",
        "hair bow",
        "ear",
        "ears",
        "human ear",
        "earlobe",
        "ear lobe",
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
        "hand",
        "palm",
        "fingers",
        "finger",
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
    debug_output_dir: Optional[str] = None,
) -> Dict[str, str]:
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]
    
    if debug_output_dir is None:
        debug_output_dir = "tmp"
    
    debug_dir = os.path.join(debug_output_dir, "debug_segmentation")
    os.makedirs(debug_dir, exist_ok=True)
    logger.info(f"Debug output directory: {debug_dir}")
    
    cv2.imwrite(os.path.join(debug_dir, "00_original.jpg"), image_bgr)
    
    results: Dict[str, str] = {}
    
    use_dino = sam3_model.backend_name == "ultralytics" and processor is not None and dino_model is not None
    
    if sam3_model.backend_name == "mlx":
        prompts_dict = BODY_PARTS_PROMPTS_CORE
    else:
        prompts_dict = BODY_PARTS_PROMPTS_FULL
    
    masks_dict: Dict[str, np.ndarray] = {}
    
    for part_name, prompts in prompts_dict.items():
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
                                prompt_mask_vis = (prompt_mask.astype(np.uint8) * 255)
                                prompt_overlay = image_bgr.copy()
                                prompt_overlay[prompt_mask] = prompt_overlay[prompt_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
                                safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:30]
                                cv2.imwrite(
                                    os.path.join(debug_dir, f"{part_name}_prompt_{i+1:02d}_{safe_prompt}_mask.png"),
                                    prompt_mask_vis
                                )
                                cv2.imwrite(
                                    os.path.join(debug_dir, f"{part_name}_prompt_{i+1:02d}_{safe_prompt}_overlay.jpg"),
                                    prompt_overlay
                                )
                                
                                if mask_total is None:
                                    mask_total = prompt_mask.copy()
                                else:
                                    mask_total |= prompt_mask
                                
                                if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                                    if mask_total is not None:
                                        merged_mask_vis = (mask_total.astype(np.uint8) * 255)
                                        merged_overlay = image_bgr.copy()
                                        merged_overlay[mask_total] = merged_overlay[mask_total] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
                                        cv2.imwrite(
                                            os.path.join(debug_dir, f"{part_name}_merged_after_prompt_{i+1:02d}_mask.png"),
                                            merged_mask_vis
                                        )
                                        cv2.imwrite(
                                            os.path.join(debug_dir, f"{part_name}_merged_after_prompt_{i+1:02d}_overlay.jpg"),
                                            merged_overlay
                                        )
                                
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
            
            mask_vis = (mask.astype(np.uint8) * 255)
            mask_overlay = image_bgr.copy()
            mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
            cv2.imwrite(os.path.join(debug_dir, f"{part_name}_before_postprocess_mask.png"), mask_vis)
            cv2.imwrite(os.path.join(debug_dir, f"{part_name}_before_postprocess_overlay.jpg"), mask_overlay)
            
            masks_dict[part_name] = mask
            
        except Exception as e:
            logger.error(f"Error segmenting {part_name}: {e}", exc_info=True)
            masks_dict[part_name] = np.zeros((h, w), dtype=bool)
    
    if "upper" in masks_dict:
        upper_mask = masks_dict["upper"].copy()
        if "head" in masks_dict:
            upper_mask = upper_mask & (~masks_dict["head"])
        if "shoes" in masks_dict:
            upper_mask = upper_mask & (~masks_dict["shoes"])
        masks_dict["upper"] = upper_mask
        upper_vis = (upper_mask.astype(np.uint8) * 255)
        upper_overlay = image_bgr.copy()
        upper_overlay[upper_mask] = upper_overlay[upper_mask] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
        cv2.imwrite(os.path.join(debug_dir, "upper_after_postprocess_mask.png"), upper_vis)
        cv2.imwrite(os.path.join(debug_dir, "upper_after_postprocess_overlay.jpg"), upper_overlay)
    
    if "lower" in masks_dict:
        lower_mask = masks_dict["lower"].copy()
        if "shoes" in masks_dict:
            lower_mask = lower_mask & (~masks_dict["shoes"])
        masks_dict["lower"] = lower_mask
        lower_vis = (lower_mask.astype(np.uint8) * 255)
        lower_overlay = image_bgr.copy()
        lower_overlay[lower_mask] = lower_overlay[lower_mask] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
        cv2.imwrite(os.path.join(debug_dir, "lower_after_postprocess_mask.png"), lower_vis)
        cv2.imwrite(os.path.join(debug_dir, "lower_after_postprocess_overlay.jpg"), lower_overlay)
    
    if "upper" in masks_dict and "lower" in masks_dict:
        upper_mask = masks_dict["upper"]
        lower_mask = masks_dict["lower"]
        overlap = upper_mask & lower_mask
        if np.any(overlap):
            upper_area = np.sum(upper_mask)
            overlap_area = np.sum(overlap)
            overlap_ratio = overlap_area / max(upper_area, 1)
            
            if overlap_ratio > 0.1:
                upper_mask = upper_mask | overlap
                lower_mask = lower_mask & (~overlap)
                masks_dict["upper"] = upper_mask
                masks_dict["lower"] = lower_mask
            else:
                upper_mask = upper_mask & (~overlap)
                lower_mask = lower_mask | overlap
                masks_dict["upper"] = upper_mask
                masks_dict["lower"] = lower_mask
            upper_final_vis = (upper_mask.astype(np.uint8) * 255)
            lower_final_vis = (lower_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(debug_dir, "upper_final_after_overlap_mask.png"), upper_final_vis)
            cv2.imwrite(os.path.join(debug_dir, "lower_final_after_overlap_mask.png"), lower_final_vis)
    
    for part_name in prompts_dict.keys():
        mask = masks_dict.get(part_name, np.zeros((h, w), dtype=bool))
        
        final_mask_vis = (mask.astype(np.uint8) * 255)
        final_overlay = image_bgr.copy()
        colors = {
            "upper": [255, 0, 0],
            "lower": [0, 0, 255],
            "shoes": [255, 255, 0],
            "head": [255, 0, 255],
            "hands": [0, 255, 255],
        }
        color = colors.get(part_name, [0, 255, 0])
        final_overlay[mask] = final_overlay[mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        cv2.imwrite(os.path.join(debug_dir, f"{part_name}_final_mask.png"), final_mask_vis)
        cv2.imwrite(os.path.join(debug_dir, f"{part_name}_final_overlay.jpg"), final_overlay)
        
        cropped_img = white_bg(image_bgr, mask)
        
        cv2.imwrite(os.path.join(debug_dir, f"{part_name}_final_cropped.jpg"), cropped_img)
        
        results[part_name] = encode_image(cropped_img, ext=".jpg")
        
        logger.info(f"Successfully segmented {part_name}")
    
    all_parts_overlay = image_bgr.copy()
    part_colors = {
        "upper": [255, 0, 0],
        "lower": [0, 0, 255],
        "shoes": [255, 255, 0],
        "head": [255, 0, 255],
        "hands": [0, 255, 255],
    }
    for part_name, color in part_colors.items():
        if part_name in masks_dict:
            mask = masks_dict[part_name]
            all_parts_overlay[mask] = all_parts_overlay[mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
    cv2.imwrite(os.path.join(debug_dir, "all_parts_merged_overlay.jpg"), all_parts_overlay)
    
    logger.info(f"Debug images saved to: {debug_dir}")
    
    return results
