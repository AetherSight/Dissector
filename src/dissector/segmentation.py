"""
身体部位分割模块
使用 SAM3 文本提示功能自动分割5个部位：upper, lower, shoes, head, hands
支持 MLX 和 Ultralytics 后端
"""
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, List
import base64
import logging
import os
import tempfile
import torch

from .backend import SAM3Base
from .constants import (
    BODY_PARTS_PROMPTS_MIX,
    BODY_PARTS_PROMPTS_ULTRA,
    DEFAULT_MIN_AREA_RATIO,
    HANDS_MIN_AREA_RATIO,
    HEAD_DILATE_KERNEL_SIZE,
    HANDS_DILATE_KERNEL_SIZE,
    MASK_CLOSE_KERNEL_SIZE,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
)
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger(__name__)

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

def mask_from_boxes(
    image_pil: Image.Image,
    boxes: np.ndarray,
    sam3_model: SAM3Base,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    save_debug: bool = False,
    debug_prefix: str = "",
    debug_dir: str = "",
) -> np.ndarray:
    """
    从边界框列表生成 mask（优化版：支持批量处理）
    
    Args:
        image_pil: PIL Image
        boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2]
        sam3_model: SAM3Base 实例
        min_area_ratio: 最小区域比例，用于过滤小组件
        save_debug: 是否保存调试图片
        debug_prefix: 调试文件前缀
        debug_dir: 调试文件目录
    
    Returns:
        Binary mask as numpy array
    """
    if boxes.size == 0:
        h, w = image_pil.size[1], image_pil.size[0]
        return np.zeros((h, w), dtype=bool)

    h, w = image_pil.size[1], image_pil.size[0]
    
    # 尝试批量处理以提高性能
    if hasattr(sam3_model, 'generate_mask_from_bboxes') and len(boxes) > 1:
        try:
            mask_total = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
            if mask_total is not None and mask_total.ndim == 2 and mask_total.shape == (h, w):
                kernel = np.ones((MASK_CLOSE_KERNEL_SIZE, MASK_CLOSE_KERNEL_SIZE), np.uint8)
                mask_total = cv2.morphologyEx(mask_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                mask_total = clean_mask(mask_total, min_area_ratio=min_area_ratio)
                return mask_total
        except Exception as e:
            logger.warning(f"[SAM3] Batch processing failed: {e}, falling back to loop")
    
    # 回退到循环处理
    mask_total = None
    individual_masks = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        bbox = np.array([[x1, y1, x2, y2]])
        
        mask = sam3_model.generate_mask_from_bbox(image_pil, bbox)
        
        if mask is None:
            continue
        
        if mask.ndim != 2:
            continue
        
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        if np.sum(mask) == 0:
            continue

        if save_debug and debug_prefix and debug_dir:
            mask_vis = (mask.astype(np.uint8)) * 255
            debug_path = os.path.join(debug_dir, f"{debug_prefix}_box_{i}_mask.png")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(debug_path, mask_vis)
            logger.info(f"Saved individual mask: {debug_path}")

        individual_masks.append(mask)
        if mask_total is None:
            mask_total = mask.copy()
        else:
            mask_total |= mask

    if mask_total is None:
        return np.zeros((h, w), dtype=bool)

    if save_debug and debug_prefix and debug_dir:
        mask_vis = (mask_total.astype(np.uint8)) * 255
        debug_path = os.path.join(debug_dir, f"{debug_prefix}_merged_mask.png")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(debug_path, mask_vis)
        logger.info(f"Saved merged mask: {debug_path}")

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    mask_total = clean_mask(mask_total, min_area_ratio=min_area_ratio)
    return mask_total


def segment_parts_mlx(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    processor=None,
    dino_model=None,
    device: Optional[torch.device] = None,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
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
    upper_mask = clean_mask(upper_mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks_dict["upper"] = upper_mask
    
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
    
    return results

def get_prompts_for_backend(backend_name: str, part_name: str):
    """
    获取后端对应的提示词
    
    Args:
        backend_name: 后端名称 ("mlx" 或 "ultralytics")
        part_name: 部位名称
    
    Returns:
        可以是 List[str]（单轮检测）或 List[List[str]]（多轮检测）
        对于 ultralytics 后端，如果 ULTRA 中有定义嵌套列表结构，则返回嵌套列表
        否则返回 MIX 的普通列表
    """
    # 对于 ultralytics 后端，优先使用 ULTRA 的提示词（可能包含嵌套列表）
    if backend_name == "ultralytics" and part_name in BODY_PARTS_PROMPTS_ULTRA:
        return BODY_PARTS_PROMPTS_ULTRA.get(part_name, [])
    
    # 其他情况使用 MIX 的提示词（普通列表）
    return BODY_PARTS_PROMPTS_MIX.get(part_name, [])


def segment_parts_ultralytics(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, str]:
    """
    Ultralytics 后端的分割函数
    采用类似MLX的策略：先检测头手腿脚，然后从整个人体mask中减去这些部位得到upper
    """
    # 延迟导入避免循环导入
    from .pipeline import run_grounding_dino
    
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]

    masks: Dict[str, np.ndarray] = {}

    def detect_and_store(key: str, prompts):
        """
        检测并存储 mask
        
        Args:
            key: 存储的键名
            prompts: 可以是以下两种格式：
                - List[str]: 单轮检测，直接使用所有提示词
                - List[List[str]]: 多轮检测，每轮检测后合并结果
        """
        # 检查 prompts 的类型
        if not prompts:
            masks[key] = np.zeros((h, w), dtype=bool)
            return
        
        # 检查第一个元素是否是列表（判断是否为嵌套列表）
        if isinstance(prompts[0], list):
            # 多轮检测：分轮执行，每轮结果合并
            mask_total = None
            for round_idx, round_prompts in enumerate(prompts):
                logger.debug(f"[ROUND {round_idx + 1}] detecting with prompts: {round_prompts}")
                dino_res = run_grounding_dino(
                    image_pil=image_pil,
                    prompts=round_prompts,
                    processor=processor,
                    dino_model=dino_model,
                    device=device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                boxes = dino_res["boxes"].cpu().numpy() if "boxes" in dino_res else np.array([])
                round_mask = mask_from_boxes(image_pil, boxes, sam3_model)
                
                # 合并到总 mask
                if mask_total is None:
                    mask_total = round_mask.copy()
                else:
                    mask_total |= round_mask
            
            masks[key] = mask_total if mask_total is not None else np.zeros((h, w), dtype=bool)
        else:
            # 单轮检测：直接使用所有提示词
            dino_res = run_grounding_dino(
                image_pil=image_pil,
                prompts=prompts,
                processor=processor,
                dino_model=dino_model,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            boxes = dino_res["boxes"].cpu().numpy() if "boxes" in dino_res else np.array([])
            masks[key] = mask_from_boxes(image_pil, boxes, sam3_model)

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
        detect_and_store("debug", ["hair", "hairband", "ear", "earring"])
        test_mask = masks.get("test_mask", np.zeros((h, w), dtype=bool))
        tmp_dir = tempfile.gettempdir()
        test_mask_vis = (test_mask.astype(np.uint8)) * 255
        debug_path = os.path.join(tmp_dir, "test_mask_debug.png")
        cv2.imwrite(debug_path, test_mask_vis)
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
    lower_negation_prompts = get_prompts_for_backend(backend_name, "lower_negation_for_upper")
    detect_and_store("lower_negation_for_upper", lower_negation_prompts)
    lower_negation_mask = masks.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))
    masks["lower_negation_for_upper"] = lower_negation_mask

    logger.debug("[STEP] detecting person/full body ...")
    person_prompts = ["person", "full body", "outfit", "garment", "human"]
    detect_and_store("person", person_prompts)
    person_mask = masks.get("person", np.zeros((h, w), dtype=bool))
    
    logger.debug("[STEP] detecting upper (direct method) ...")
    detect_and_store("upper_raw", get_prompts_for_backend(backend_name, "upper"))
    
    person_mask = masks.get("person", np.zeros((h, w), dtype=bool))
    upper_detected = masks.get("upper_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    lower_negation_mask = masks.get("lower_negation_for_upper", np.zeros((h, w), dtype=bool))
    
    upper_detected_cleaned = upper_detected.copy()
    upper_detected_cleaned = upper_detected_cleaned & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_detected_cleaned = upper_detected_cleaned & (~masks[part_name])
    
    logger.debug("[STEP] computing upper from person mask subtraction ...")
    upper_from_subtraction = person_mask.copy()
    upper_from_subtraction = upper_from_subtraction & (~lower_negation_mask)
    for part_name in ["shoes", "head", "hands"]:
        if part_name in masks:
            upper_from_subtraction = upper_from_subtraction & (~masks[part_name])
    
    upper_mask = upper_detected_cleaned | upper_from_subtraction
    upper_mask = clean_mask(upper_mask, min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    masks["upper"] = upper_mask
    
    masks["shoes"] = clean_mask(masks.get("shoes", np.zeros_like(upper_mask)), min_area_ratio=DEFAULT_MIN_AREA_RATIO)
    
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
    return results


def segment_parts(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    processor=None,
    dino_model=None,
    device: Optional[torch.device] = None,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> Dict[str, str]:
    """统一的分割入口，根据后端类型调用相应的实现"""
    if sam3_model.backend_name == "mlx":
        return segment_parts_mlx(
            image_pil=image_pil,
            sam3_model=sam3_model,
            processor=processor,
            dino_model=dino_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    else:
        return segment_parts_ultralytics(
            image_pil=image_pil,
            sam3_model=sam3_model,
            processor=processor,
            dino_model=dino_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
