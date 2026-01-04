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
import tempfile

from .sam3_backend import SAM3Base

logger = logging.getLogger(__name__)

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
    ],
    "head": [
        "head hair accessory",
        "head accessory",
        "hair accessory",
        "hair flower",
        "flower hair accessory",
        "headwear",
        "headpiece",
        "headdress",
        "hat",
        "ear",
        "ears",
        "human ear",
        "earlobe",
        "ear lobe",
        "ear accessory",
        "earring",
        "earrings",
        "ear jewelry",
        "cap",
        "helmet",
        "crown",
        "tiara",
        "headband",
        "hood",
        "hair ornament",
        "hair clip",
        "hairpin",
        "hair band",
        "hair ribbon",
        "hair bow",
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
            
            # 统一使用文本提示词叠加方式（MLX 和 Ultralytics 都尝试）
            # 分别调用每个提示词，然后合并结果（避免超长提示词被截断）
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
                    logger.warning(f"Error with prompt '{prompt}': {e}")
                    continue
            
            mask = mask_total
            
            # 如果文本提示词方式失败且是 Ultralytics，回退到 DINO + SAM3
            if mask is None or mask.size == 0:
                if use_dino:
                    logger.info(f"Text prompt failed for {part_name}, falling back to DINO + SAM3")
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
                        # 使用 SAM3 从边界框生成 mask
                        mask = sam3_model.generate_mask_from_bboxes(image_pil, boxes)
                    else:
                        logger.warning(f"No boxes detected for {part_name}")
            
            if mask is None or mask.size == 0:
                logger.warning(f"No mask found for {part_name}")
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
        # 保存原始检测的 mask（后处理前）
        upper_raw = masks_dict["upper"].copy()
        debug_dir = tempfile.gettempdir()
        if np.any(upper_raw):
            upper_raw_vis = (upper_raw.astype(np.uint8) * 255)
            upper_raw_path = os.path.join(debug_dir, "upper_mask_raw.png")
            cv2.imwrite(upper_raw_path, upper_raw_vis)
            logger.info(f"[DEBUG] Saved upper raw mask to: {upper_raw_path}")
        
        upper_mask = masks_dict["upper"].copy()
        if "head" in masks_dict:
            upper_mask = upper_mask & (~masks_dict["head"])
        if "shoes" in masks_dict:
            upper_mask = upper_mask & (~masks_dict["shoes"])
        
        # 保存排除 head 和 shoes 后的 mask
        if np.any(upper_mask):
            upper_after_exclude_vis = (upper_mask.astype(np.uint8) * 255)
            upper_after_exclude_path = os.path.join(debug_dir, "upper_mask_after_exclude.png")
            cv2.imwrite(upper_after_exclude_path, upper_after_exclude_vis)
            logger.info(f"[DEBUG] Saved upper mask after excluding head/shoes to: {upper_after_exclude_path}")
        
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
            
            # 保存重叠处理前的 mask
            debug_dir = tempfile.gettempdir()
            upper_before_overlap_vis = (upper_mask.astype(np.uint8) * 255)
            upper_before_overlap_path = os.path.join(debug_dir, "upper_mask_before_overlap.png")
            cv2.imwrite(upper_before_overlap_path, upper_before_overlap_vis)
            logger.info(f"[DEBUG] Saved upper mask before overlap handling to: {upper_before_overlap_path}")
            
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
    
    # 临时调试：保存 upper 的最终结果
    if "upper" in masks_dict:
        debug_dir = tempfile.gettempdir()
        upper_final = masks_dict["upper"]
        
        if np.any(upper_final):
            # 保存最终 mask
            upper_final_vis = (upper_final.astype(np.uint8) * 255)
            upper_final_path = os.path.join(debug_dir, "upper_mask_final.png")
            cv2.imwrite(upper_final_path, upper_final_vis)
            logger.info(f"[DEBUG] Saved upper final mask to: {upper_final_path}")
            
            # 保存最终抠图
            upper_final_cropped = render_white_bg(image_bgr, upper_final)
            upper_final_cropped_path = os.path.join(debug_dir, "upper_cropped_final.png")
            cv2.imwrite(upper_final_cropped_path, upper_final_cropped)
            logger.info(f"[DEBUG] Saved upper final cropped image to: {upper_final_cropped_path}")
            
            # 保存叠加效果
            upper_overlay = image_bgr.copy()
            overlay = upper_overlay.copy()
            overlay[upper_final] = [0, 255, 0]  # 绿色叠加
            upper_overlay = cv2.addWeighted(upper_overlay, 0.7, overlay, 0.3, 0)
            upper_overlay_path = os.path.join(debug_dir, "upper_overlay_final.png")
            cv2.imwrite(upper_overlay_path, upper_overlay)
            logger.info(f"[DEBUG] Saved upper final overlay to: {upper_overlay_path}")
    
    # 生成最终结果
    for part_name in BODY_PARTS_PROMPTS.keys():
        mask = masks_dict.get(part_name, np.zeros((h, w), dtype=bool))
        
        # 生成白色背景的抠图
        cropped_img = render_white_bg(image_bgr, mask)
        
        # 编码为 base64
        results[part_name] = encode_bgr_to_base64(cropped_img, ext=".jpg")
        
        logger.info(f"Successfully segmented {part_name}")
    
    return results

