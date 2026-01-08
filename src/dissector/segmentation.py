"""
Body parts segmentation module.
Uses SAM3 text prompts to automatically segment 5 parts: upper, lower, shoes, head, hands.
Supports MLX and CUDA (Facebook SAM3) backends.
"""
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional
import base64
import logging
import os
import tempfile
import torch

from .backend import SAM3Base
from .constants import (
    BODY_PARTS_PROMPTS,
    DEFAULT_MIN_AREA_RATIO,
    HANDS_MIN_AREA_RATIO,
    HEAD_DILATE_KERNEL_SIZE,
    HANDS_DILATE_KERNEL_SIZE,
    MASK_CLOSE_KERNEL_SIZE,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Hide DEBUG output from segmentation module

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

def clean_mask(mask: np.ndarray, min_area_ratio: float = DEFAULT_MIN_AREA_RATIO) -> np.ndarray:
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

def segment_parts_mlx(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    device: Optional[torch.device] = None,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> Dict[str, str]:
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]
    
    results: Dict[str, str] = {}
    prompts_dict = BODY_PARTS_PROMPTS
    masks_dict: Dict[str, np.ndarray] = {}
    other_parts = ["lower", "shoes", "head", "hands"]
    
    for part_name, prompts in prompts_dict.items():
        try:
            mask_total = None
            for prompt_idx, prompt in enumerate(prompts):
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
            logger.info("lower_negation_for_upper prompts returned no mask (legs/pants may be covered), using empty mask")
            lower_negation_mask = np.zeros((h, w), dtype=bool)
        else:
            lower_negation_mask = lower_negation_mask_total
            if lower_negation_mask.shape != (h, w):
                mask_uint8 = (lower_negation_mask.astype(np.uint8) * 255) if lower_negation_mask.dtype == bool else lower_negation_mask.astype(np.uint8)
                lower_negation_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        masks_dict["lower_negation_for_upper"] = lower_negation_mask
    else:
        logger.info("lower_negation_for_upper not in prompts, using empty mask")
        masks_dict["lower_negation_for_upper"] = np.zeros((h, w), dtype=bool)
    
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
    lower_mask_for_upper = masks_dict.get("lower", np.zeros((h, w), dtype=bool))
    
    person_minus_extremities = person_mask.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            person_minus_extremities = person_minus_extremities & (~masks_dict[part_name])
    if person_minus_extremities.shape != (h, w):
        mask_uint8 = (person_minus_extremities.astype(np.uint8) * 255) if person_minus_extremities.dtype == bool else person_minus_extremities.astype(np.uint8)
        person_minus_extremities = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    upper_original = upper_detected.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_original = upper_original & (~masks_dict[part_name])
    if upper_original.shape != (h, w):
        mask_uint8 = (upper_original.astype(np.uint8) * 255) if upper_original.dtype == bool else upper_original.astype(np.uint8)
        upper_original = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    upper_original = clean_mask(upper_original, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper"] = upper_original
    
    upper_1 = upper_detected.copy()
    upper_1 = upper_1 & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_1 = upper_1 & (~masks_dict[part_name])
    upper_1 = clean_mask(upper_1, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper_1"] = upper_1
    
    upper_2 = upper_detected.copy()
    upper_2 = upper_2 & (~lower_mask_for_upper)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_2 = upper_2 & (~masks_dict[part_name])
    upper_2 = clean_mask(upper_2, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper_2"] = upper_2
    
    upper_3 = person_minus_extremities.copy()
    upper_3 = upper_3 & (~lower_mask_for_upper)
    upper_3 = clean_mask(upper_3, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper_3"] = upper_3
    
    upper_4 = person_minus_extremities.copy()
    upper_4 = upper_4 & (~lower_negation_mask)
    upper_4 = clean_mask(upper_4, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper_4"] = upper_4
    
    if "lower" in masks_dict:
        lower_mask = masks_dict["lower"].copy()
        if "shoes" in masks_dict:
            lower_mask = lower_mask & (~masks_dict["shoes"])
        lower_mask = clean_mask(lower_mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
        masks_dict["lower"] = lower_mask
    
    for part_name in prompts_dict.keys():
        mask = masks_dict.get(part_name, np.zeros((h, w), dtype=bool))
        if part_name not in ["upper", "lower"]:
            mask = clean_mask(mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
        cropped_img = white_bg(image_bgr, mask)
        results[part_name] = encode_image(cropped_img, ext=".jpg")
    
    for variant_name in ["upper_1", "upper_2", "upper_3", "upper_4"]:
        if variant_name in masks_dict:
            mask = masks_dict[variant_name]
            cropped_img = white_bg(image_bgr, mask)
            results[variant_name] = encode_image(cropped_img, ext=".jpg")
    
    return results

def debug_get_mask(
    part_name: str,
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    debug_dir: str = "./tmp",
) -> None:
    prompts_dict = BODY_PARTS_PROMPTS
    if part_name not in prompts_dict:
        logger.warning(f"Part '{part_name}' not found in prompts dict")
        return
    
    prompts = prompts_dict[part_name]
    h, w = image_pil.size[1], image_pil.size[0]
    
    os.makedirs(debug_dir, exist_ok=True)
    
    logger.info(f"Debug: generating masks for part '{part_name}' with {len(prompts)} prompts")
    
    mask_total = None
    valid_prompts_count = 0
    
    for prompt_idx, prompt in enumerate(prompts):
        try:
            logger.debug(f"Processing prompt {prompt_idx+1}/{len(prompts)}: '{prompt}'")
            prompt_mask = sam3_model.generate_mask_from_text_prompt(
                image_pil=image_pil,
                text_prompt=prompt,
            )
            
            if prompt_mask is None:
                logger.warning(f"Prompt '{prompt}' returned None mask")
                continue
            
            if prompt_mask.size == 0:
                logger.warning(f"Prompt '{prompt}' returned empty mask")
                continue
            
            mask_pixels = np.sum(prompt_mask)
            if mask_pixels == 0:
                logger.warning(f"Prompt '{prompt}' generated mask with 0 pixels")
                continue
            
            if prompt_mask.shape != (h, w):
                mask_uint8 = (prompt_mask.astype(np.uint8) * 255) if prompt_mask.dtype == bool else prompt_mask.astype(np.uint8)
                mask_vis = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                prompt_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask_vis = (prompt_mask.astype(np.uint8)) * 255
                if prompt_mask.dtype != bool:
                    prompt_mask = prompt_mask.astype(bool)
            
            prompt_safe = prompt.replace(" ", "_").replace("/", "_")
            debug_path = os.path.join(debug_dir, f"{part_name}_prompt_{prompt_idx+1:02d}_{prompt_safe}.png")
            
            success = cv2.imwrite(debug_path, mask_vis)
            if success:
                logger.info(f"Saved debug mask: {debug_path} (pixels: {mask_pixels})")
                valid_prompts_count += 1
            else:
                logger.error(f"Failed to save debug mask: {debug_path}")
                continue
            
            if mask_total is None:
                mask_total = prompt_mask.copy()
            else:
                mask_total |= prompt_mask
                
        except Exception as e:
            logger.warning(f"Failed to generate mask for prompt '{prompt}': {e}", exc_info=True)
            continue
    
    if mask_total is not None and np.sum(mask_total) > 0:
        mask_total_vis = (mask_total.astype(np.uint8)) * 255
        merged_path = os.path.join(debug_dir, f"{part_name}_merged_mask.png")
        success = cv2.imwrite(merged_path, mask_total_vis)
        if success:
            total_pixels = np.sum(mask_total)
            logger.info(f"Saved merged mask: {merged_path} (total pixels: {total_pixels}, from {valid_prompts_count} prompts)")
        else:
            logger.error(f"Failed to save merged mask: {merged_path}")
        
        try:
            image_rgb = np.array(image_pil)
            overlay = image_rgb.copy()
            color_map = {
                "upper": [0, 255, 0],  # Green
                "lower": [255, 0, 0],  # Blue
                "shoes": [0, 0, 255],  # Red
                "head": [255, 255, 0],  # Cyan
                "hands": [255, 0, 255],  # Magenta
                "lower_negation_for_upper": [255, 165, 0],  # Orange
            }
            color = color_map.get(part_name, [0, 255, 0])
            overlay[mask_total] = overlay[mask_total] * 0.7 + np.array(color) * 0.3
            overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
            overlay_path = os.path.join(debug_dir, f"{part_name}_merged_overlay.jpg")
            success = cv2.imwrite(overlay_path, overlay_bgr)
            if success:
                logger.info(f"Saved merged overlay: {overlay_path}")
            else:
                logger.error(f"Failed to save merged overlay: {overlay_path}")
        except Exception as e:
            logger.warning(f"Failed to create overlay image: {e}")
    else:
        if part_name == "lower_negation_for_upper":
            logger.warning(f"No valid masks generated for part '{part_name}' from prompts")
            logger.warning("Note: In actual segmentation, this will use empty mask (no fallback)")
            if "lower" in BODY_PARTS_PROMPTS:
                logger.info("You may want to debug 'lower' part to see what mask would be used as reference")
        else:
            logger.warning(f"No valid masks generated for part '{part_name}', skipping merged mask")


def get_prompts_for_backend(backend_name: str, part_name: str):
    """
    Get prompts for the specified backend.
    
    Args:
        backend_name: Backend name ("mlx" or "cuda")
        part_name: Part name
    
    Returns:
        Can be List[str] (single-round detection) or List[List[str]] (multi-round detection)
    """
    return BODY_PARTS_PROMPTS.get(part_name, [])


def segment_parts_cuda(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    device: Optional[torch.device] = None,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> Dict[str, str]:
    """
    CUDA (Facebook SAM3) backend segmentation function.
    Uses SAM3 text prompts for semantic segmentation, consistent with MLX backend.
    """
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]

    masks: Dict[str, np.ndarray] = {}
    prompts_dict = BODY_PARTS_PROMPTS
    other_parts = ["lower", "shoes", "head", "hands"]

    def detect_and_store(key: str, prompts):
        """
        Detect and store mask using SAM3 text prompts.
        
        Args:
            key: Storage key name
            prompts: Can be in one of two formats:
                - List[str]: Single-round detection, use all prompts directly
                - List[List[str]]: Multi-round detection, merge results after each round
        """
        if not prompts:
            masks[key] = np.zeros((h, w), dtype=bool)
            return
        
        if isinstance(prompts[0], list):
            mask_total = None
            for round_idx, round_prompts in enumerate(prompts):
                logger.debug(f"[ROUND {round_idx + 1}] detecting with prompts: {round_prompts}")
                round_mask_total = None
                for prompt in round_prompts:
                    try:
                        prompt_mask = sam3_model.generate_mask_from_text_prompt(
                            image_pil=image_pil,
                            text_prompt=prompt,
                        )
                        if prompt_mask is not None and prompt_mask.size > 0 and np.sum(prompt_mask) > 0:
                            if prompt_mask.shape != (h, w):
                                mask_uint8 = (prompt_mask.astype(np.uint8) * 255) if prompt_mask.dtype == bool else prompt_mask.astype(np.uint8)
                                prompt_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            else:
                                if prompt_mask.dtype != bool:
                                    prompt_mask = prompt_mask.astype(bool)
                            
                            if round_mask_total is None:
                                round_mask_total = prompt_mask.copy()
                            else:
                                round_mask_total |= prompt_mask
                    except Exception as e:
                        logger.warning(f"Failed to generate mask for prompt '{prompt}': {e}")
                        continue
                
                if round_mask_total is not None:
                    if mask_total is None:
                        mask_total = round_mask_total.copy()
                    else:
                        mask_total |= round_mask_total
            
            masks[key] = mask_total if mask_total is not None else np.zeros((h, w), dtype=bool)
        else:
            mask_total = None
            for prompt in prompts:
                try:
                    prompt_mask = sam3_model.generate_mask_from_text_prompt(
                        image_pil=image_pil,
                        text_prompt=prompt,
                    )
                    if prompt_mask is not None and prompt_mask.size > 0 and np.sum(prompt_mask) > 0:
                        if prompt_mask.shape != (h, w):
                            mask_uint8 = (prompt_mask.astype(np.uint8) * 255) if prompt_mask.dtype == bool else prompt_mask.astype(np.uint8)
                            prompt_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        else:
                            if prompt_mask.dtype != bool:
                                prompt_mask = prompt_mask.astype(bool)
                        
                        if mask_total is None:
                            mask_total = prompt_mask.copy()
                        else:
                            mask_total |= prompt_mask
                except Exception as e:
                    logger.warning(f"Failed to generate mask for prompt '{prompt}': {e}")
                    continue
            
            masks[key] = mask_total if mask_total is not None else np.zeros((h, w), dtype=bool)

    backend_name = sam3_model.backend_name

    logger.debug("[STEP] detecting shoes ...")
    detect_and_store("shoes", get_prompts_for_backend(backend_name, "shoes"))

    logger.debug("[STEP] detecting lower ...")
    detect_and_store("lower_raw", get_prompts_for_backend(backend_name, "lower"))
    lower_mask = masks.get("lower_raw", np.zeros((h, w), dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    masks["lower"] = clean_mask(lower_mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)

    logger.debug("[STEP] detecting head ...")
    detect_and_store("head", get_prompts_for_backend(backend_name, "head"))
    head_mask = masks.get("head", np.zeros((h, w), dtype=bool))
    if np.any(head_mask):
        kernel = np.ones((HEAD_DILATE_KERNEL_SIZE, HEAD_DILATE_KERNEL_SIZE), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["head"] = head_mask
    
    if False:
        detect_and_store("debug", ["triangular ears"])
        debug_mask = masks.get("debug", np.zeros((h, w), dtype=bool))
        tmp_dir = tempfile.gettempdir()
        debug_mask_vis = (debug_mask.astype(np.uint8)) * 255
        debug_path = os.path.join(tmp_dir, "debug_mask.png")
        cv2.imwrite(debug_path, debug_mask_vis)
        logger.info(f"Saved test-only mask debug image: {debug_path}")

    logger.debug("[STEP] detecting hands ...")
    detect_and_store("hands", get_prompts_for_backend(backend_name, "hands"))
    hand_mask = masks.get("hands", np.zeros(image_rgb.shape[:2], dtype=bool))
    hand_mask = clean_mask(hand_mask, min_area_ratio=HANDS_MIN_AREA_RATIO)
    if np.any(hand_mask):
        kernel = np.ones((HANDS_DILATE_KERNEL_SIZE, HANDS_DILATE_KERNEL_SIZE), np.uint8)
        hand_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["hands"] = hand_mask

    logger.debug("[STEP] detecting lower_negation_for_upper ...")
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
            logger.info("lower_negation_for_upper prompts returned no mask (legs/pants may be covered), using empty mask")
            lower_negation_mask = np.zeros((h, w), dtype=bool)
        else:
            lower_negation_mask = lower_negation_mask_total
            if lower_negation_mask.shape != (h, w):
                mask_uint8 = (lower_negation_mask.astype(np.uint8) * 255) if lower_negation_mask.dtype == bool else lower_negation_mask.astype(np.uint8)
                lower_negation_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        masks["lower_negation_for_upper"] = lower_negation_mask
    else:
        logger.info("lower_negation_for_upper not in prompts, using empty mask")
        masks["lower_negation_for_upper"] = np.zeros((h, w), dtype=bool)
    
    lower_negation_mask = masks.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))

    logger.debug("[STEP] detecting person/full body ...")
    person_prompts = ["person", "full body", "outfit", "garment", "human"]
    detect_and_store("person", person_prompts)
    person_mask = masks.get("person", np.zeros((h, w), dtype=bool))
    
    logger.debug("[STEP] detecting upper (direct method) ...")
    detect_and_store("upper_raw", get_prompts_for_backend(backend_name, "upper"))
    
    person_mask = masks.get("person", np.zeros((h, w), dtype=bool))
    upper_detected = masks.get("upper_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    lower_negation_mask = masks.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))
    lower_mask_for_upper = masks.get("lower", np.zeros((h, w), dtype=bool))
    
    person_minus_extremities = person_mask.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            person_minus_extremities = person_minus_extremities & (~masks[part_name])
    
    upper_original = upper_detected.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_original = upper_original & (~masks[part_name])
    
    upper_original = clean_mask(upper_original, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    
    masks["upper"] = upper_original
    
    upper_1 = upper_detected.copy()
    upper_1 = upper_1 & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_1 = upper_1 & (~masks[part_name])
    upper_1 = clean_mask(upper_1, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["upper_1"] = upper_1
    
    upper_2 = upper_detected.copy()
    upper_2 = upper_2 & (~lower_mask_for_upper)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_2 = upper_2 & (~masks[part_name])
    upper_2 = clean_mask(upper_2, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["upper_2"] = upper_2
    
    upper_3 = person_minus_extremities.copy()
    upper_3 = upper_3 & (~lower_mask_for_upper)
    upper_3 = clean_mask(upper_3, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["upper_3"] = upper_3
    
    upper_4 = person_minus_extremities.copy()
    upper_4 = upper_4 & (~lower_negation_mask)
    upper_4 = clean_mask(upper_4, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["upper_4"] = upper_4
    
    masks["shoes"] = clean_mask(masks.get("shoes", np.zeros((h, w), dtype=bool)), min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    
    results: Dict[str, str] = {}
    outputs = [
        ("upper", "upper"),
        ("lower", "lower"),
        ("shoes", "shoes"),
        ("head", "head"),
        ("hands", "hands"),
    ]
    for key, name in outputs:
        mask_part = masks.get(key, np.zeros((h, w), dtype=bool))
        out_img = white_bg(image_bgr, mask_part)
        results[name] = encode_image(out_img, ext=".jpg")
    
    for variant_name in ["upper_1", "upper_2", "upper_3", "upper_4"]:
        if variant_name in masks:
            mask = masks[variant_name]
            cropped_img = white_bg(image_bgr, mask)
            results[variant_name] = encode_image(cropped_img, ext=".jpg")
    
    return results


def segment_parts(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    device: Optional[torch.device] = None,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> Dict[str, str]:
    """Unified segmentation entry point, calls corresponding implementation based on backend type."""
    if sam3_model.backend_name == "mlx":
        return segment_parts_mlx(
            image_pil=image_pil,
            sam3_model=sam3_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    else:
        return segment_parts_cuda(
            image_pil=image_pil,
            sam3_model=sam3_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
