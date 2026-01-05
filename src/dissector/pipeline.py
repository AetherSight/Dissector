import os
import base64
import logging
import platform
import time
import threading
from typing import Dict, List, Tuple, Any, Optional

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
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .backend import SAM3Factory, SAM3Base
from .segmentation import (
    segment_parts,
    BODY_PARTS_PROMPTS_CORE,
    BODY_PARTS_PROMPTS_FULL,
)

def get_prompts_for_backend(backend_name: str, part_name: str) -> List[str]:
    if backend_name == "mlx":
        return BODY_PARTS_PROMPTS_CORE.get(part_name, [])
    else:
        return BODY_PARTS_PROMPTS_FULL.get(part_name, [])

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

LEG_PROMPTS: List[str] = [
    "leg",
    "legs",
    "human leg",
    "thigh",
    "thighs",
]

def load_models(
    dino_model_name: str,
    device: torch.device,
    sam3_backend: Optional[str] = None,
) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, SAM3Base]:
    """
    Load models for image processing.
    
    Args:
        dino_model_name: Grounding DINO model name
        device: PyTorch device
        sam3_backend: SAM3 backend type ("mlx" or "ultralytics"), None for auto-detect
    
    Returns:
        Tuple of (processor, dino_model, sam3_model)
    """
    processor = AutoProcessor.from_pretrained(
        dino_model_name,
        local_files_only=True,
    )
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_model_name,
        local_files_only=True,
    ).to(device)
    
    device_str = None
    if device.type == "cuda":
        device_str = "cuda"
    elif device.type == "mps":
        device_str = "mps"
    else:
        device_str = "cpu"
    
    sam3_model = SAM3Factory.create(backend=sam3_backend, device=device_str)
    logger.info(f"Loaded SAM3 model with backend: {sam3_model.backend_name}")
    
    return processor, dino_model, sam3_model


def run_grounding_dino(
    image_pil: Image.Image,
    prompts: List[str],
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> Dict:
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
    return results


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


def mask_from_boxes(
    image_pil: Image.Image,
    boxes: np.ndarray,
    sam3_model: SAM3Base,
    min_area_ratio: float = 0.001,
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
                mask_total = clean_mask(mask_total, min_area_ratio=min_area_ratio)
                logger.info(f"[SAM3] Batch processing succeeded for {len(boxes)} boxes")
                return mask_total
            else:
                logger.warning(f"[SAM3] Batch processing returned invalid mask (shape: {mask_total.shape if mask_total is not None else None}), falling back to loop")
        except Exception as e:
            logger.warning(f"[SAM3] Batch processing failed: {e}, falling back to loop", exc_info=True)
    elif len(boxes) > 1:
        logger.warning(f"[SAM3] Batch processing not available (hasattr: {hasattr(sam3_model, 'generate_mask_from_bboxes')}), using loop for {len(boxes)} boxes")
    
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


def white_bg(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def encode_image(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def remove_background(
    image_path: str,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam3_model: SAM3Base,
    device: torch.device,
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
    ]
    
    dino_res = run_grounding_dino(
        image_pil=image_pil,
        prompts=person_prompts,
        processor=processor,
        dino_model=dino_model,
        device=device,
        box_threshold=0.25,
        text_threshold=0.2,
    )
    boxes = dino_res["boxes"].cpu().numpy() if "boxes" in dino_res else np.array([])
    
    if boxes.size == 0:
        person_mask = np.ones((h, w), dtype=bool)
    else:
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
            
            if mask_total is None:
                mask_total = mask.copy()
            else:
                mask_total |= mask
        
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
    out = white_bg(image_bgr, mask)
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
    processor: Optional[AutoProcessor] = None,
    dino_model: Optional[AutoModelForZeroShotObjectDetection] = None,
    device: Optional[torch.device] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Dict[str, str]:
    """
    简化的图片处理函数：传入图片，返回5个部位的抠图（base64编码）
    
    Args:
        image_pil: PIL Image 对象（RGB格式）
        sam3_model: SAM3Base 实例
        processor: DINO processor（仅 Ultralytics 需要）
        dino_model: DINO model（仅 Ultralytics 需要）
        device: PyTorch device（仅 Ultralytics 需要）
        box_threshold: DINO 框阈值（仅 Ultralytics 需要）
        text_threshold: DINO 文本阈值（仅 Ultralytics 需要）
    
    Returns:
        字典，包含5个部位的 base64 编码图片
    """
    return segment_parts(
        image_pil=image_pil,
        sam3_model=sam3_model,
        processor=processor,
        dino_model=dino_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )


def process_image(
    image_path: str,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam3_model: SAM3Base,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, str]:
    start_time = time.time()
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] cannot read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    if sam3_model.backend_name == "mlx":
        logger.info("[PERF] MLX backend: using segment_parts (no DINO)")
        segment_start = time.time()
        results = segment_parts(
            image_pil=image_pil,
            sam3_model=sam3_model,
            processor=None,
            dino_model=None,
            device=None,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        segment_time = time.time() - segment_start
        logger.info(f"[PERF] MLX segment_parts: {segment_time:.2f}s")
        return results

    masks: Dict[str, np.ndarray] = {}

    # 创建 tmp 目录用于保存中间 mask
    tmp_dir = os.path.join(os.path.dirname(image_path), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    def detect_and_store(key: str, prompts: List[str]):
        """原本的 detect_and_store 函数，用于 ultralytics 后端"""
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
        masks[key] = mask_from_boxes(image_pil, boxes, sam3_model, save_debug=True, debug_prefix=f"{base_name}_{key}", debug_dir=tmp_dir)

    # Ultralytics 后端：按照原本的顺序处理
    # 注意：MLX 后端已在上面通过 segment_parts 处理并返回
    logger.info("[STEP] detecting shoes ...")
    # 使用原本的 FOOTWEAR_PROMPTS，顺序和内容必须与原本代码完全一致
    FOOTWEAR_PROMPTS: List[str] = [
        "shoe",
        "shoes",
        "boot",
        "boots",
        "sandal",
        "sneaker",
        "high heel",
        "flat shoe",
    ]
    detect_and_store("shoes", FOOTWEAR_PROMPTS)

    logger.info("[STEP] detecting lower ...")
    # 使用原本的 LOWER_PROMPTS，顺序和内容必须与原本代码完全一致
    LOWER_PROMPTS: List[str] = [
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
    ]
    detect_and_store("lower_raw", LOWER_PROMPTS)
    lower_mask = masks.get("lower_raw", np.zeros((h, w), dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    masks["lower"] = clean_mask(lower_mask, min_area_ratio=0.001)

    logger.info("[STEP] detecting head (for removal) ...")
    # 使用原本的 HEADWEAR_PROMPTS，顺序和内容必须与原本代码完全一致
    HEADWEAR_PROMPTS: List[str] = [
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
    ]
    detect_and_store("head", HEADWEAR_PROMPTS)
    head_mask = masks.get("head", np.zeros((h, w), dtype=bool))
    if np.any(head_mask):
        kernel = np.ones((15, 15), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["head"] = head_mask

    logger.info("[STEP] detecting upper ...")
    # 使用原本的 UPPER_PROMPTS，顺序和内容必须与原本代码完全一致
    UPPER_PROMPTS: List[str] = [
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
    ]
    detect_and_store("upper_raw", UPPER_PROMPTS)
    upper_mask = masks.get("upper_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    upper_mask = (
        upper_mask
        & (~masks.get("lower", np.zeros_like(upper_mask)))
        & (~masks.get("shoes", np.zeros_like(upper_mask)))
        & (~masks.get("head", np.zeros_like(upper_mask)))
    )
    upper_mask = clean_mask(upper_mask, min_area_ratio=0.001)
    masks["upper"] = upper_mask
    masks["shoes"] = clean_mask(masks.get("shoes", np.zeros_like(upper_mask)), min_area_ratio=0.001)

    logger.info("[STEP] detecting hands (remove from upper)...")
    # 使用原本的 HAND_PROMPTS，顺序和内容必须与原本代码完全一致
    HAND_PROMPTS: List[str] = [
        "human hand",
        "hands",
        "palm",
        "fingers",
        "bare hand",
        "bare fingers",
        "gloves",
        "ring",
        "hand accessory",
    ]
    detect_and_store("hands", HAND_PROMPTS)
    hand_mask = masks.get("hands", np.zeros(image_rgb.shape[:2], dtype=bool))
    hand_mask = clean_mask(hand_mask, min_area_ratio=0.0005)
    if np.any(hand_mask):
        kernel = np.ones((5, 5), np.uint8)
        hand_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["hands"] = hand_mask

    masks["upper"] = masks.get("upper", np.zeros_like(hand_mask)) & (~hand_mask)
    masks["upper"] = clean_mask(masks["upper"], min_area_ratio=0.001)
    
    # 保存每个部位的最终 mask 到 tmp 目录
    for key in ["shoes", "lower_raw", "lower", "head", "upper_raw", "upper", "hands"]:
        if key in masks:
            mask_vis = (masks[key].astype(np.uint8)) * 255
            mask_path = os.path.join(tmp_dir, f"{base_name}_{key}_final_mask.png")
            cv2.imwrite(mask_path, mask_vis)
            logger.info(f"Saved final mask: {mask_path}")
    
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
        out_img = white_bg(image_bgr, mask_part)
        results[name] = encode_image(out_img, ext=".jpg")
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

