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
import io
import torch
import os

from .sam3_backend import SAM3Base

logger = logging.getLogger(__name__)

# 估算 token 数量的辅助函数（简单方法：1 token ≈ 0.75 单词，或约 4 字符）
def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量
    使用简单启发式：单词数 * 1.33 或 字符数 / 4，取较大值
    """
    word_count = len(text.split())
    char_count = len(text)
    # 取两种估算方法的平均值，向上取整
    tokens_by_words = int(word_count * 1.33) + 1
    tokens_by_chars = int(char_count / 4) + 1
    return max(tokens_by_words, tokens_by_chars)

def batch_prompts_by_token_limit(prompts: List[str], max_tokens: int = 64) -> List[str]:
    """
    将提示词列表按 token 限制分组，每组用逗号连接
    
    Args:
        prompts: 提示词列表
        max_tokens: 每组最大 token 数（默认 64）
    
    Returns:
        分组后的提示词列表，每组用逗号连接
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    for prompt in prompts:
        prompt_tokens = estimate_tokens(prompt)
        
        # 如果单个提示词就超过限制，单独成组
        if prompt_tokens > max_tokens:
            if current_batch:
                batches.append(", ".join(current_batch))
                current_batch = []
                current_tokens = 0
            batches.append(prompt)
            continue
        
        # 检查加入当前提示词后是否超过限制
        # 需要加上分隔符的 token（", " 约 1 token）
        separator_tokens = 1 if current_batch else 0
        if current_tokens + separator_tokens + prompt_tokens > max_tokens:
            # 当前批次已满，保存并开始新批次
            if current_batch:
                batches.append(", ".join(current_batch))
            current_batch = [prompt]
            current_tokens = prompt_tokens
        else:
            # 加入当前批次
            current_batch.append(prompt)
            current_tokens += separator_tokens + prompt_tokens
    
    # 处理最后一个批次
    if current_batch:
        batches.append(", ".join(current_batch))
    
    return batches

# 5个部位的文本提示词
BODY_PARTS_PROMPTS = {
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
        "belt",
        "waistband",
        "waist belt",
        "garment body",
        "clothing fabric",
        "inner lining",
        "lining",
        "inner fabric",
        "garment lining",
        "clothing lining",
    ],
    "lower": [
        # 核心下装（优先处理）
        "pants",
        "trousers",
        "jeans",
        "slacks",
        "shorts",
        "leggings",
        "tights",
        "pant legs",
        "trouser legs",
        # 内衬
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
    ],
    "head": [
        # 核心头部和脸部（优先处理）
        "head",
        "human head",
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
        "hair",
        "hairstyle",
        "ponytail hair",
        # 头部配饰
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
        # 头发配饰
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
        # 耳部
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
        "human hand",
        "hands",
        "palm",
        "fingers",
        "bare hand",
        "bare fingers",
    ],
}


def render_white_bg(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    将 mask 区域外的背景渲染为白色
    
    Args:
        image_bgr: BGR 格式的图片，shape (H, W, 3)
        mask: 二进制 mask，shape (H, W), dtype=bool
    
    Returns:
        白色背景的图片，BGR 格式
    """
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


def encode_bgr_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    """
    将 BGR 图片编码为 base64 字符串
    
    Args:
        img_bgr: BGR 格式的图片
        ext: 图片格式扩展名
    
    Returns:
        base64 编码的字符串
    """
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def remove_small_components(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """
    移除 mask 中的小组件
    
    Args:
        mask: 二进制 mask
        min_area_ratio: 最小区域比例
    
    Returns:
        清理后的 mask
    """
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


def run_grounding_dino_for_prompts(
    image_pil: Image.Image,
    prompts: List[str],
    processor,
    dino_model,
    device: torch.device,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> np.ndarray:
    """
    使用 Grounding DINO 检测并返回边界框
    
    Args:
        image_pil: PIL Image
        prompts: 提示词列表
        processor: DINO processor
        dino_model: DINO model
        device: PyTorch device
        box_threshold: 框阈值
        text_threshold: 文本阈值
    
    Returns:
        边界框数组，shape (N, 4) with [x1, y1, x2, y2]
    """
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


def segment_body_parts_with_sam3(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
    processor=None,
    dino_model=None,
    device: Optional[torch.device] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Dict[str, str]:
    """
    使用 SAM3 模型分割5个身体部位并返回抠图结果（base64编码）
    
    Args:
        image_pil: PIL Image 对象（RGB格式）
        sam3_model: SAM3Base 实例（MLX 或 Ultralytics）
        processor: DINO processor（仅 Ultralytics 需要）
        dino_model: DINO model（仅 Ultralytics 需要）
        device: PyTorch device（仅 Ultralytics 需要）
        box_threshold: DINO 框阈值（仅 Ultralytics 需要）
        text_threshold: DINO 文本阈值（仅 Ultralytics 需要）
    
    Returns:
        字典，包含5个部位的 base64 编码图片：
        {
            "upper": base64_string,
            "lower": base64_string,
            "shoes": base64_string,
            "head": base64_string,
            "hands": base64_string
        }
    """
    # 转换 PIL Image 为 numpy array (BGR格式用于 OpenCV)
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]
    
    results: Dict[str, str] = {}
    
    # 检查是否使用 Ultralytics（需要 DINO）
    use_dino = sam3_model.backend_name == "ultralytics" and processor is not None and dino_model is not None
    
    # 存储所有部位的 mask，用于后处理
    masks_dict: Dict[str, np.ndarray] = {}
    
    # 对每个部位进行分割
    for part_name, prompts in BODY_PARTS_PROMPTS.items():
        try:
            logger.info(f"Segmenting {part_name}...")
            
            mask = None
            
            # 根据后端类型选择处理方式
            if sam3_model.backend_name == "mlx":
                # MLX: 批量发送提示词（每组最多 64 token）
                batched_prompts = batch_prompts_by_token_limit(prompts, max_tokens=64)
                logger.info(f"MLX: {part_name} prompts batched into {len(batched_prompts)} groups")
                
                mask_total = None
                successful_batches = 0
                failed_batches = 0
                
                for i, batch_prompt in enumerate(batched_prompts):
                    try:
                        batch_mask = sam3_model.generate_mask_from_text_prompt(
                            image_pil=image_pil,
                            text_prompt=batch_prompt,
                        )
                        if batch_mask is not None and batch_mask.size > 0:
                            mask_pixels = np.sum(batch_mask)
                            if mask_pixels > 0:
                                if mask_total is None:
                                    mask_total = batch_mask.copy()
                                else:
                                    mask_total |= batch_mask
                                successful_batches += 1
                                logger.debug(f"MLX: {part_name} batch {i+1}/{len(batched_prompts)} succeeded, mask pixels: {mask_pixels}")
                            else:
                                failed_batches += 1
                                logger.debug(f"MLX: {part_name} batch {i+1}/{len(batched_prompts)} returned empty mask")
                        else:
                            failed_batches += 1
                            logger.debug(f"MLX: {part_name} batch {i+1}/{len(batched_prompts)} returned None or empty")
                    except Exception as e:
                        failed_batches += 1
                        logger.warning(f"MLX: {part_name} batch {i+1}/{len(batched_prompts)} error with prompt '{batch_prompt[:50]}...': {e}")
                        continue
                
                if mask_total is not None and mask_total.size > 0:
                    total_pixels = np.sum(mask_total)
                    logger.info(f"MLX: {part_name} completed: {successful_batches} successful, {failed_batches} failed batches, total mask pixels: {total_pixels}")
                else:
                    logger.warning(f"MLX: {part_name} failed: no valid mask generated from {len(batched_prompts)} batches")
                
                mask = mask_total
                
                # 如果 MLX 文本提示词失败，尝试 DINO 回退（如果可用）
                if (mask is None or mask.size == 0) and use_dino:
                    logger.info(f"MLX text prompt failed for {part_name}, falling back to DINO + SAM3")
                    boxes = run_grounding_dino_for_prompts(
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
                # Ultralytics: 直接使用 DINO + SAM3（不支持文本提示词）
                if use_dino:
                    logger.info(f"Using DINO + SAM3 for {part_name} (Ultralytics backend)")
                    boxes = run_grounding_dino_for_prompts(
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
                # 存储空 mask
                masks_dict[part_name] = np.zeros((h, w), dtype=bool)
                continue
            
            # 确保 mask 是正确的形状和类型
            if mask.shape != (h, w):
                mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
                mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # 清理 mask（暂时注释掉，测试是否影响头饰检测）
            # mask = remove_small_components(mask, min_area_ratio=0.001)
            
            # 存储 mask 用于后处理
            masks_dict[part_name] = mask
            
        except Exception as e:
            logger.error(f"Error segmenting {part_name}: {e}", exc_info=True)
            # 存储空 mask
            masks_dict[part_name] = np.zeros((h, w), dtype=bool)
    
    # 后处理：排除重叠区域
    # upper 排除 head 和 shoes
    if "upper" in masks_dict:
        upper_mask = masks_dict["upper"].copy()
        if "head" in masks_dict:
            upper_mask = upper_mask & (~masks_dict["head"])
        if "shoes" in masks_dict:
            upper_mask = upper_mask & (~masks_dict["shoes"])
        # 暂时注释掉，测试是否影响头饰检测
        # masks_dict["upper"] = remove_small_components(upper_mask, min_area_ratio=0.001)
        masks_dict["upper"] = upper_mask
    
    # lower 排除 shoes
    if "lower" in masks_dict:
        lower_mask = masks_dict["lower"].copy()
        if "shoes" in masks_dict:
            lower_mask = lower_mask & (~masks_dict["shoes"])
        # 暂时注释掉，测试是否影响头饰检测
        # masks_dict["lower"] = remove_small_components(lower_mask, min_area_ratio=0.001)
        masks_dict["lower"] = lower_mask
    
    # upper 和 lower 的重叠处理
    if "upper" in masks_dict and "lower" in masks_dict:
        upper_mask = masks_dict["upper"]
        lower_mask = masks_dict["lower"]
        overlap = upper_mask & lower_mask
        if np.any(overlap):
            upper_area = np.sum(upper_mask)
            overlap_area = np.sum(overlap)
            overlap_ratio = overlap_area / max(upper_area, 1)
            
            if overlap_ratio > 0.1:
                # 重叠区域归 upper
                upper_mask = upper_mask | overlap
                lower_mask = lower_mask & (~overlap)
                # 暂时注释掉，测试是否影响头饰检测
                # masks_dict["upper"] = remove_small_components(upper_mask, min_area_ratio=0.001)
                # masks_dict["lower"] = remove_small_components(lower_mask, min_area_ratio=0.001)
                masks_dict["upper"] = upper_mask
                masks_dict["lower"] = lower_mask
            else:
                # 重叠区域归 lower
                upper_mask = upper_mask & (~overlap)
                lower_mask = lower_mask | overlap
                # 暂时注释掉，测试是否影响头饰检测
                # masks_dict["upper"] = remove_small_components(upper_mask, min_area_ratio=0.001)
                # masks_dict["lower"] = remove_small_components(lower_mask, min_area_ratio=0.001)
                masks_dict["upper"] = upper_mask
                masks_dict["lower"] = lower_mask
    
    # 生成最终结果
    for part_name in BODY_PARTS_PROMPTS.keys():
        mask = masks_dict.get(part_name, np.zeros((h, w), dtype=bool))
        
        # 生成白色背景的抠图
        cropped_img = render_white_bg(image_bgr, mask)
        
        # 编码为 base64
        results[part_name] = encode_bgr_to_base64(cropped_img, ext=".jpg")
        
        logger.info(f"Successfully segmented {part_name}")
    
    return results

