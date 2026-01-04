import os
import base64
import logging
import platform
import time
import threading
from typing import Dict, List, Tuple, Any, Optional
# Parallel processing disabled to avoid SIGSEGV and resource leaks
# from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("transformers.image_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
from PIL import Image

from .sam3_backend import SAM3Factory, SAM3Base
from .body_parts_segmentation import segment_body_parts_with_sam3, BODY_PARTS_PROMPTS


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    On macOS with MLX backend, use CPU to avoid MPS concurrency issues.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif platform.system() == "Darwin":
        return torch.device("cpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 从 body_parts_segmentation 导入统一的提示词源
HEADWEAR_PROMPTS = BODY_PARTS_PROMPTS["head"]
UPPER_PROMPTS = BODY_PARTS_PROMPTS["upper"]
LOWER_PROMPTS = BODY_PARTS_PROMPTS["lower"]
FOOTWEAR_PROMPTS = BODY_PARTS_PROMPTS["shoes"]
HAND_PROMPTS = BODY_PARTS_PROMPTS["hands"]
LEG_PROMPTS: List[str] = [
    "leg",
    "legs",
    "human leg",
    "thigh",
    "thighs",
]

def load_models(
    device: torch.device,
    sam3_backend: Optional[str] = None,
) -> SAM3Base:
    """
    Load models for image processing.
    
    Args:
        device: PyTorch device (not used for MLX, used for Ultralytics device selection)
        sam3_backend: SAM3 backend type ("mlx" or "ultralytics"), None for auto-detect
    
    Returns:
        SAM3Base instance
    """
    # 自动检测设备类型用于 Ultralytics
    device_str = None
    if device.type == "cuda":
        device_str = "cuda"
    elif device.type == "mps":
        device_str = "mps"
    else:
        device_str = "cpu"
    
    sam3_model = SAM3Factory.create(backend=sam3_backend, device=device_str)
    logger.info(f"Loaded SAM3 model with backend: {sam3_model.backend_name}")
    
    return sam3_model


def remove_small_components(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
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
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    """
    从边界框列表生成 mask（优化版：支持批量处理）
    
    Args:
        image_pil: PIL Image
        boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2]
        sam3_model: SAM3Base 实例
        min_area_ratio: 最小区域比例，用于过滤小组件
    
    Returns:
        Binary mask as numpy array
    """
    if boxes.size == 0:
        h, w = image_pil.size[1], image_pil.size[0]
        return np.zeros((h, w), dtype=bool)

    h, w = image_pil.size[1], image_pil.size[0]
    
    if hasattr(sam3_model, 'generate_mask_from_bboxes') and len(boxes) > 1:
        logger.info(f"[SAM3] Attempting batch processing for {len(boxes)} boxes")
        try:
            mask_total = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
            if mask_total is not None and mask_total.ndim == 2 and mask_total.shape == (h, w):
                kernel = np.ones((3, 3), np.uint8)
                mask_total = cv2.morphologyEx(mask_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                mask_total = remove_small_components(mask_total, min_area_ratio=min_area_ratio)
                logger.info(f"[SAM3] Batch processing succeeded for {len(boxes)} boxes")
                return mask_total
            else:
                logger.warning(f"[SAM3] Batch processing returned invalid mask (shape: {mask_total.shape if mask_total is not None else None}), falling back to loop")
        except Exception as e:
            logger.warning(f"[SAM3] Batch processing failed: {e}, falling back to loop", exc_info=True)
    elif len(boxes) > 1:
        logger.warning(f"[SAM3] Batch processing not available (hasattr: {hasattr(sam3_model, 'generate_mask_from_bboxes')}), using loop for {len(boxes)} boxes")
    
    mask_total = None
    for box in boxes:
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

        if mask_total is None:
            mask_total = mask.copy()
        else:
            mask_total |= mask

    if mask_total is None:
        return np.zeros((h, w), dtype=bool)

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    mask_total = remove_small_components(mask_total, min_area_ratio=min_area_ratio)
    return mask_total


def render_white_bg(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.full_like(image_bgr, 255, dtype=np.uint8)

    mask_uint8 = (mask.astype(np.uint8)) * 255
    ys, xs = np.where(mask_uint8 > 0)
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


def encode_bgr_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def remove_background(
    image_path: str,
    sam3_model: SAM3Base,
) -> str:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]
    
    person_prompts = [
        "person",
        "human",
        "human body",
    ]
    
    # 使用文本提示词叠加方式
    mask_total = None
    for prompt in person_prompts:
        try:
            single_mask = sam3_model.generate_mask_from_text_prompt(
                image_pil=image_pil,
                text_prompt=prompt,
            )
            if single_mask is not None and single_mask.size > 0:
                if mask_total is None:
                    mask_total = single_mask.copy()
                else:
                    mask_total |= single_mask
        except Exception as e:
            logger.warning(f"Error with prompt '{prompt}': {e}")
            continue
    
    if mask_total is None or np.sum(mask_total) == 0:
        person_mask = np.ones((h, w), dtype=bool)
    else:
        person_mask = mask_total
    
    image_rgba = image_rgb.copy()
    alpha_channel = (person_mask.astype(np.uint8) * 255)
    image_rgba = np.dstack([image_rgba, alpha_channel])
    
    image_pil_rgba = Image.fromarray(image_rgba, "RGBA")
    
    import io
    buffer = io.BytesIO()
    image_pil_rgba.save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_with_white_bg(image_bgr: np.ndarray, mask: np.ndarray, output_path: str):
    out = render_white_bg(image_bgr, mask)
    if out is None or out.size == 0:
        logger.warning(f"mask is empty, skip save: {output_path}")
        return
    cv2.imwrite(output_path, out)
    logger.info(f"saved: {output_path}")


def save_debug_overlay(image_bgr: np.ndarray, mask: np.ndarray, output_path: str, color: Tuple[int, int, int], title: str):
    vis = image_bgr.copy()
    overlay = vis.copy()
    overlay[mask] = color
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imwrite(output_path, vis)


def process_image_simple(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
) -> Dict[str, str]:
    """
    简化的图片处理函数：传入图片，返回5个部位的抠图（base64编码）
    
    Args:
        image_pil: PIL Image 对象（RGB格式）
        sam3_model: SAM3Base 实例
    
    Returns:
        字典，包含5个部位的 base64 编码图片
    """
    return segment_body_parts_with_sam3(
        image_pil=image_pil,
        sam3_model=sam3_model,
    )


def process_image(
    image_path: str,
    sam3_model: SAM3Base,
) -> Dict[str, str]:
    start_time = time.time()
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] cannot read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    masks: Dict[str, np.ndarray] = {}

    def process_sam3_with_text_prompts(key: str, prompts: List[str]):
        """使用文本提示词叠加方式处理"""
        step_start = time.time()
        mask_total = None
        for prompt in prompts:
            try:
                single_mask = sam3_model.generate_mask_from_text_prompt(
                    image_pil=image_pil,
                    text_prompt=prompt,
                )
                if single_mask is not None and single_mask.size > 0:
                    if mask_total is None:
                        mask_total = single_mask.copy()
                    else:
                        mask_total |= single_mask
            except Exception as e:
                logger.warning(f"Error with prompt '{prompt}' for {key}: {e}")
                continue
        
        if mask_total is not None:
            logger.info(f"[PERF] {key}: SAM3={time.time()-step_start:.2f}s, prompts={len(prompts)}")
            return key, mask_total
        else:
            logger.warning(f"No mask generated for {key}")
            return key, np.zeros((h, w), dtype=bool)

    logger.info("[STEP] Stage 1: detecting shoes, head, lower_raw, upper_raw (sequential)...")
    stage1_start = time.time()
    
    # 统一使用文本提示词叠加方式
    for key, prompts in [("shoes", FOOTWEAR_PROMPTS), ("head", HEADWEAR_PROMPTS), 
                          ("lower_raw", LOWER_PROMPTS), ("upper_raw", UPPER_PROMPTS)]:
        try:
            _, mask = process_sam3_with_text_prompts(key, prompts)
            masks[key] = mask
        except Exception as e:
            logger.error(f"[PERF] Stage 1 task {key} failed: {e}", exc_info=True)
            masks[key] = np.zeros((h, w), dtype=bool)
    
    if sam3_model.backend_name == "mlx":
        try:
            import mlx.core as mx
            mx.eval()
            try:
                mx.metal.clear_cache()
                logger.debug("[MLX] Cleared Metal cache after Stage 1")
            except AttributeError:
                pass
            logger.debug("[MLX] Triggered lazy evaluation and cleared cache for Stage 1")
        except ImportError:
            pass
    
    logger.info(f"[PERF] Stage 1 completed: {time.time()-stage1_start:.2f}s")
    
    head_mask = masks.get("head", np.zeros((h, w), dtype=bool))
    
    if np.any(head_mask):
        kernel = np.ones((15, 15), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["head"] = head_mask
    
    lower_mask = masks.get("lower_raw", np.zeros((h, w), dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    masks["lower"] = remove_small_components(lower_mask, min_area_ratio=0.001)
    upper_mask = masks.get("upper_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    lower_mask_current = masks.get("lower", np.zeros_like(upper_mask))
    
    # 临时调试：保存 upper 的原始检测 mask（后处理前）
    if np.any(upper_mask):
        import tempfile
        import os
        debug_dir = tempfile.gettempdir()
        upper_raw_vis = (upper_mask.astype(np.uint8) * 255)
        upper_raw_path = os.path.join(debug_dir, "upper_mask_raw.png")
        cv2.imwrite(upper_raw_path, upper_raw_vis)
        logger.info(f"[DEBUG] Saved upper raw mask to: {upper_raw_path}")
    
    upper_mask = (
        upper_mask
        & (~masks.get("shoes", np.zeros_like(upper_mask)))
        & (~masks.get("head", np.zeros_like(upper_mask)))
    )
    
    # 临时调试：保存排除 head 和 shoes 后的 mask
    if np.any(upper_mask):
        import tempfile
        import os
        debug_dir = tempfile.gettempdir()
        upper_after_exclude_vis = (upper_mask.astype(np.uint8) * 255)
        upper_after_exclude_path = os.path.join(debug_dir, "upper_mask_after_exclude.png")
        cv2.imwrite(upper_after_exclude_path, upper_after_exclude_vis)
        logger.info(f"[DEBUG] Saved upper mask after excluding head/shoes to: {upper_after_exclude_path}")
    
    if np.any(upper_mask) and np.any(lower_mask_current):
        overlap = upper_mask & lower_mask_current
        if np.any(overlap):
            upper_area = np.sum(upper_mask)
            overlap_area = np.sum(overlap)
            overlap_ratio = overlap_area / max(upper_area, 1)
            
            if overlap_ratio > 0.1:
                upper_mask = upper_mask | overlap
                lower_mask_current = lower_mask_current & (~overlap)
                masks["lower"] = lower_mask_current
    
    upper_mask = upper_mask & (~lower_mask_current)
    
    # 临时调试：保存处理重叠后的 mask（清理前）
    if np.any(upper_mask):
        import tempfile
        import os
        debug_dir = tempfile.gettempdir()
        upper_before_clean_vis = (upper_mask.astype(np.uint8) * 255)
        upper_before_clean_path = os.path.join(debug_dir, "upper_mask_before_clean.png")
        cv2.imwrite(upper_before_clean_path, upper_before_clean_vis)
        logger.info(f"[DEBUG] Saved upper mask before cleaning to: {upper_before_clean_path}")
    
    upper_mask = remove_small_components(upper_mask, min_area_ratio=0.001)
    masks["upper"] = upper_mask
    masks["shoes"] = remove_small_components(masks.get("shoes", np.zeros_like(upper_mask)), min_area_ratio=0.001)
    
    # 临时调试：保存 upper 的最终结果
    if np.any(upper_mask):
        import tempfile
        import os
        debug_dir = tempfile.gettempdir()
        upper_final_vis = (upper_mask.astype(np.uint8) * 255)
        upper_final_path = os.path.join(debug_dir, "upper_mask_final.png")
        cv2.imwrite(upper_final_path, upper_final_vis)
        logger.info(f"[DEBUG] Saved upper final mask to: {upper_final_path}")
        
        # 保存最终抠图
        upper_final_cropped = render_white_bg(image_bgr, upper_mask)
        upper_final_cropped_path = os.path.join(debug_dir, "upper_cropped_final.png")
        cv2.imwrite(upper_final_cropped_path, upper_final_cropped)
        logger.info(f"[DEBUG] Saved upper final cropped image to: {upper_final_cropped_path}")
        
        # 保存叠加效果
        upper_overlay = image_bgr.copy()
        overlay = upper_overlay.copy()
        overlay[upper_mask] = [0, 255, 0]  # 绿色叠加
        upper_overlay = cv2.addWeighted(upper_overlay, 0.7, overlay, 0.3, 0)
        upper_overlay_path = os.path.join(debug_dir, "upper_overlay_final.png")
        cv2.imwrite(upper_overlay_path, upper_overlay)
        logger.info(f"[DEBUG] Saved upper final overlay to: {upper_overlay_path}")

    logger.info("[STEP] Stage 2: detecting legs and hands (sequential)...")
    stage2_start = time.time()
    
    # 统一使用文本提示词叠加方式
    for key, prompts in [("legs", LEG_PROMPTS), ("hands", HAND_PROMPTS)]:
        try:
            _, mask = process_sam3_with_text_prompts(key, prompts)
            masks[key] = mask
        except Exception as e:
            logger.error(f"[PERF] Stage 2 task {key} failed: {e}", exc_info=True)
            masks[key] = np.zeros((h, w), dtype=bool)
    
    if sam3_model.backend_name == "mlx":
        try:
            import mlx.core as mx
            mx.eval()
            try:
                mx.metal.clear_cache()
                logger.debug("[MLX] Cleared Metal cache after Stage 2")
            except AttributeError:
                pass
            logger.debug("[MLX] Triggered lazy evaluation for Stage 2")
        except ImportError:
            pass
    
    logger.info(f"[PERF] Stage 2 completed: {time.time()-stage2_start:.2f}s")
    
    leg_mask = masks.get("legs", np.zeros(image_rgb.shape[:2], dtype=bool))
    leg_mask = remove_small_components(leg_mask, min_area_ratio=0.0005)
    
    if np.any(leg_mask) and np.any(upper_mask):
        leg_in_upper = leg_mask & upper_mask
        if np.any(leg_in_upper):
            upper_mask = upper_mask & (~leg_in_upper)
            masks["lower"] = masks.get("lower", np.zeros_like(leg_in_upper)) | leg_in_upper
            masks["lower"] = remove_small_components(masks["lower"], min_area_ratio=0.001)
            upper_mask = remove_small_components(upper_mask, min_area_ratio=0.001)
            masks["upper"] = upper_mask
    
    hand_mask = masks.get("hands", np.zeros(image_rgb.shape[:2], dtype=bool))
    hand_mask = remove_small_components(hand_mask, min_area_ratio=0.0005)
    
    if np.any(hand_mask) and np.any(masks.get("lower", np.zeros_like(hand_mask))):
        lower_mask = masks.get("lower", np.zeros_like(hand_mask))
        overlap = hand_mask & lower_mask
        if np.any(overlap):
            overlap_ratio = np.sum(overlap) / max(np.sum(hand_mask), 1)
            if overlap_ratio > 0.5:
                logger.info(f"Hand mask mostly overlaps with lower ({overlap_ratio:.2%}), excluding overlap")
                hand_mask = hand_mask & (~overlap)
    
    if np.any(hand_mask):
        kernel = np.ones((5, 5), np.uint8)
        hand_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["hands"] = hand_mask

    masks["upper"] = masks.get("upper", np.zeros_like(hand_mask)) & (~hand_mask)
    masks["upper"] = remove_small_components(masks["upper"], min_area_ratio=0.001)
    results: Dict[str, str] = {}
    outputs = [
        ("upper", "upper"),
        ("lower", "lower"),
        ("shoes", "shoes"),
        ("head", "head"),
        ("hands", "hands"),
    ]
    step_start = time.time()
    for key, name in outputs:
        mask_part = masks.get(key, np.zeros((h, w), dtype=bool))
        out_img = render_white_bg(image_bgr, mask_part)
        results[name] = encode_bgr_to_base64(out_img, ext=".jpg")
    encode_time = time.time() - step_start
    
    total_time = time.time() - start_time
    logger.info(f"[PERF] Total: {total_time:.2f}s (encode: {encode_time:.2f}s)")
    return results


def run_batch(
    input_dir: str,
    output_dir: str,
    dino_model_name: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> List[Dict[str, str]]:
    device = get_device()
    logger.info(f"device: {device} (platform: {platform.system()})")
    logger.info("loading models ...")
    processor, dino_model, sam3_model = load_models(dino_model_name, device)
    logger.info("models loaded.")

    images = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    logger.info(f"found {len(images)} images.")

    results_all: List[Dict[str, str]] = []
    for img in images:
        logger.info(f"processing {os.path.basename(img)}")
        res = process_image(
            image_path=img,
            processor=processor,
            dino_model=dino_model,
            sam3_model=sam3_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        results_all.append(res)
        logger.info(f"done {os.path.basename(img)}")

    return results_all

