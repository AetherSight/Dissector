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


def close_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological close if configured."""
    if MASK_CLOSE_KERNEL_SIZE and MASK_CLOSE_KERNEL_SIZE > 1:
        k = np.ones((MASK_CLOSE_KERNEL_SIZE, MASK_CLOSE_KERNEL_SIZE), np.uint8)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return mask


def build_person_mask(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    h: int,
    w: int,
) -> np.ndarray:
    person_prompts = ["person", "full body", "outfit", "garment", "human", "accessory"]
    mask_total = None
    for prompt in person_prompts:
        try:
            prompt_mask = sam3_model.generate_mask_from_text_prompt(
                image_pil=image_pil,
                text_prompt=prompt,
            )
            if prompt_mask is None or prompt_mask.size == 0 or np.sum(prompt_mask) == 0:
                continue
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
        except Exception:
            continue
    if mask_total is None or np.sum(mask_total) == 0:
        # Fallback to full image to避免空结果
        return np.ones((h, w), dtype=bool)
    return mask_total.astype(bool)



def get_roi_from_mask(mask: np.ndarray, pad: int = 10) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    return y0, y1 + 1, x0, x1 + 1

def segment_parts_mlx(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
) -> Dict[str, str]:
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]
    
    results: Dict[str, str] = {}
    prompts_dict = BODY_PARTS_PROMPTS
    masks_dict: Dict[str, np.ndarray] = {}

    # 1) 获取人物掩码并裁剪 ROI
    person_mask = build_person_mask(image_pil=image_pil, sam3_model=sam3_model, h=h, w=w)

    y0, y1, x0, x1 = get_roi_from_mask(person_mask)
    roi_rgb = image_rgb[y0:y1, x0:x1]
    roi_pil = Image.fromarray(roi_rgb)
    roi_h, roi_w = roi_rgb.shape[:2]

    def detect_on_roi(prompts) -> np.ndarray:
        if not prompts:
            return np.zeros((h, w), dtype=bool)
        mask_total = None
        for prompt in prompts:
            try:
                prompt_mask = sam3_model.generate_mask_from_text_prompt(
                    image_pil=roi_pil,
                    text_prompt=prompt,
                )
                if prompt_mask is None or prompt_mask.size == 0 or np.sum(prompt_mask) == 0:
                    continue
                if prompt_mask.shape != (roi_h, roi_w):
                    mask_uint8 = (prompt_mask.astype(np.uint8) * 255) if prompt_mask.dtype == bool else prompt_mask.astype(np.uint8)
                    prompt_mask = cv2.resize(mask_uint8, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    if prompt_mask.dtype != bool:
                        prompt_mask = prompt_mask.astype(bool)
                if mask_total is None:
                    mask_total = prompt_mask.copy()
                else:
                    mask_total |= prompt_mask
            except Exception:
                continue
        full_mask = np.zeros((h, w), dtype=bool)
        if mask_total is not None and np.sum(mask_total) > 0:
            full_mask[y0:y1, x0:x1] = mask_total
        return full_mask

    # 2) 逐部位检测（基于 ROI）
    for part_name, prompts in prompts_dict.items():
        masks_dict[part_name] = detect_on_roi(prompts)

    # 补充 lower_negation_for_upper 已包含在 prompts_dict 上面
    lower_negation_mask = masks_dict.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))

    # 3) 形态学与清理
    for key in list(masks_dict.keys()):
        masks_dict[key] = close_mask(masks_dict[key])

    # 4) 组合与过滤
    masks_dict["shoes"] = clean_mask(masks_dict.get("shoes", np.zeros((h, w), dtype=bool)), min_area_ratio=DEFAULT_MIN_AREA_RATIO)

    lower_mask = masks_dict.get("lower", np.zeros((h, w), dtype=bool))
    lower_mask = lower_mask & (~masks_dict.get("shoes", np.zeros_like(lower_mask)))
    lower_mask = clean_mask(lower_mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["lower"] = lower_mask

    head_mask = masks_dict.get("head", np.zeros((h, w), dtype=bool))
    if np.any(head_mask):
        kernel = np.ones((HEAD_DILATE_KERNEL_SIZE, HEAD_DILATE_KERNEL_SIZE), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks_dict["head"] = head_mask

    hand_mask = masks_dict.get("hands", np.zeros((h, w), dtype=bool))
    hand_mask = clean_mask(hand_mask, min_area_ratio=HANDS_MIN_AREA_RATIO)
    if np.any(hand_mask):
        kernel = np.ones((HANDS_DILATE_KERNEL_SIZE, HANDS_DILATE_KERNEL_SIZE), np.uint8)
        hand_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks_dict["hands"] = hand_mask

    person_minus_extremities = person_mask.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            person_minus_extremities = person_minus_extremities & (~masks_dict[part_name])

    upper_detected = masks_dict.get("upper", np.zeros((h, w), dtype=bool))
    lower_mask_for_upper = masks_dict.get("lower", np.zeros((h, w), dtype=bool))

    upper_original = upper_detected.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_original = upper_original & (~masks_dict[part_name])
    upper_original = upper_original & person_mask
    upper_original = clean_mask(upper_original, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper"] = upper_original

    upper_1 = upper_detected.copy()
    upper_1 = upper_1 & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_1 = upper_1 & (~masks_dict[part_name])
    upper_1 = upper_1 & person_mask
    upper_1 = clean_mask(upper_1, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper_1"] = upper_1

    upper_2 = upper_detected.copy()
    upper_2 = upper_2 & (~lower_mask_for_upper)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks_dict:
            upper_2 = upper_2 & (~masks_dict[part_name])
    upper_2 = upper_2 & person_mask
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

    # 5) 输出
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
    backend_name = sam3_model.backend_name

    # 1) 人物掩码与 ROI
    person_mask = build_person_mask(image_pil=image_pil, sam3_model=sam3_model, h=h, w=w)
    y0, y1, x0, x1 = get_roi_from_mask(person_mask)
    roi_rgb = image_rgb[y0:y1, x0:x1]
    roi_pil = Image.fromarray(roi_rgb)
    roi_h, roi_w = roi_rgb.shape[:2]

    def detect_and_store(key: str, prompts):
        if not prompts:
            masks[key] = np.zeros((h, w), dtype=bool)
            return
        mask_total = None
        # 支持多轮提示
        if isinstance(prompts[0], list):
            prompt_rounds = prompts
        else:
            prompt_rounds = [prompts]
        for round_prompts in prompt_rounds:
            round_mask_total = None
            for prompt in round_prompts:
                try:
                    prompt_mask = sam3_model.generate_mask_from_text_prompt(
                        image_pil=roi_pil,
                        text_prompt=prompt,
                    )
                    if prompt_mask is None or prompt_mask.size == 0 or np.sum(prompt_mask) == 0:
                        continue
                    if prompt_mask.shape != (roi_h, roi_w):
                        mask_uint8 = (prompt_mask.astype(np.uint8) * 255) if prompt_mask.dtype == bool else prompt_mask.astype(np.uint8)
                        prompt_mask = cv2.resize(mask_uint8, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST).astype(bool)
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
        full_mask = np.zeros((h, w), dtype=bool)
        if mask_total is not None and np.sum(mask_total) > 0:
            full_mask[y0:y1, x0:x1] = mask_total
        masks[key] = full_mask

    # 2) 部位检测（基于 ROI）
    detect_and_store("shoes", get_prompts_for_backend(backend_name, "shoes"))
    detect_and_store("lower_raw", get_prompts_for_backend(backend_name, "lower"))
    detect_and_store("head", get_prompts_for_backend(backend_name, "head"))
    detect_and_store("hands", get_prompts_for_backend(backend_name, "hands"))
    detect_and_store("lower_negation_for_upper", prompts_dict.get("lower_negation_for_upper", []))
    detect_and_store("upper_raw", get_prompts_for_backend(backend_name, "upper"))

    # 3) 形态学与清理
    for key in list(masks.keys()):
        masks[key] = close_mask(masks[key])

    lower_negation_mask = masks.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))

    lower_mask = masks.get("lower_raw", np.zeros((h, w), dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    lower_mask = clean_mask(lower_mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["lower"] = lower_mask

    head_mask = masks.get("head", np.zeros((h, w), dtype=bool))
    if np.any(head_mask):
        kernel = np.ones((HEAD_DILATE_KERNEL_SIZE, HEAD_DILATE_KERNEL_SIZE), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["head"] = head_mask

    hand_mask = masks.get("hands", np.zeros(image_rgb.shape[:2], dtype=bool))
    hand_mask = clean_mask(hand_mask, min_area_ratio=HANDS_MIN_AREA_RATIO)
    if np.any(hand_mask):
        kernel = np.ones((HANDS_DILATE_KERNEL_SIZE, HANDS_DILATE_KERNEL_SIZE), np.uint8)
        hand_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["hands"] = hand_mask

    person_minus_extremities = person_mask.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            person_minus_extremities = person_minus_extremities & (~masks[part_name])

    upper_detected = masks.get("upper_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    lower_mask_for_upper = masks.get("lower", np.zeros((h, w), dtype=bool))

    upper_original = upper_detected.copy()
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_original = upper_original & (~masks[part_name])
    upper_original = upper_original & person_mask
    upper_original = clean_mask(upper_original, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    
    masks["upper"] = upper_original
    
    upper_1 = upper_detected.copy()
    upper_1 = upper_1 & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_1 = upper_1 & (~masks[part_name])
    upper_1 = upper_1 & person_mask
    upper_1 = clean_mask(upper_1, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["upper_1"] = upper_1
    
    upper_2 = upper_detected.copy()
    upper_2 = upper_2 & (~lower_mask_for_upper)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_2 = upper_2 & (~masks[part_name])
    upper_2 = upper_2 & person_mask
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
) -> Dict[str, str]:
    """Unified segmentation entry point, calls corresponding implementation based on backend type."""
    if sam3_model.backend_name == "mlx":
        return segment_parts_mlx(
            image_pil=image_pil,
            sam3_model=sam3_model,
        )
    else:
        return segment_parts_cuda(
            image_pil=image_pil,
            sam3_model=sam3_model,
        )
