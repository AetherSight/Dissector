import os
import argparse
from typing import List, Dict, Tuple

import hydra
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Prompt words for detecting different body parts
HEADGEAR_PARTS: List[str] = [
    "hat",  # 帽子
    "cap",  # 鸭舌帽
    "helmet",  # 头盔
    "headgear",  # 头饰
    "crown",  # 王冠
    "tiara",  # 头冠
    "headband",  # 头带
    "hood",  # 兜帽
]

UPPER_GARMENT_PARTS: List[str] = [
    # Main garment types - emphasize full garment body
    "dress",  # 连衣裙
    "gown",  # 长袍/礼服
    "robe",  # 长袍
    "tunic",  # 束腰外衣
    "shirt",  # 衬衫
    "blouse",  # 女式上衣
    "jacket",  # 夹克
    "coat",  # 外套
    "sweater",  # 毛衣
    "hoodie",  # 连帽衫
    "vest",  # 背胸
    "tank top",  # 背心
    "t-shirt",  # T恤
    "top",  # 上装
    "upper garment",  # 上衣
    "upper body clothing",  # 上半身服装
    "chest piece",  # 胸甲
    "chest armor",  # 胸甲
    "breastplate",  # 胸甲板
    # Garment body structure
    "garment body",  # 衣服主体
    "clothing fabric",  # 服装面料
    "dress bodice",  # 连衣裙上身
    "dress top",  # 连衣裙上身部分
    "upper part of dress",  # 连衣裙的上半部分
]

LOWER_GARMENT_PARTS: List[str] = [
    # Specific lower garment types - emphasize pants/skirt only
    "pants",  # 裤子
    "trousers",  # 长裤
    "shorts",  # 短裤
    "skirt",  # 裙子
    "pant legs",  # 裤腿
    "trouser legs",  # 裤腿
    "pant leg",  # 单条裤腿
    "trouser leg",  # 单条裤腿
    "leg covering",  # 腿部覆盖物
    "waistband",  # 腰头
    "pant waist",  # 裤腰
    "pant crotch",  # 裤裆
    "pant inseam",  # 裤内缝
]

FOOTWEAR_PARTS: List[str] = [
    "shoe",  # 鞋子
    "boot",  # 靴子
    "sandal",  # 凉鞋
    "sneaker",  # 运动鞋
    "footwear",  # 鞋类
    "boots",  # 靴子
    "high heel",  # 高跟鞋
    "flat shoe",  # 平底鞋
]

GLOVES_PARTS: List[str] = [
    "glove",  # 手套
    "gloves",  # 手套
    "gauntlet",  # 护手
    "hand guard",  # 手部护具
    "wrist guard",  # 护腕
]

# Prompt words for detecting upper body accessories (arm bands, chest accessories, etc.)
UPPER_ACCESSORY_PARTS: List[str] = [
    # Arm accessories
    "arm band",  # 臂环/臂章
    "armlet",  # 臂环
    "bracer",  # 护腕
    "wristband",  # 腕带
    "arm accessory",  # 手臂配饰
    # Chest and upper body accessories
    "chest pocket",  # 胸袋
    "breast pocket",  # 胸前口袋
    "chest bag",  # 胸包
    "chest pouch",  # 胸前小包
    # Shoulder accessories
    "shoulder pad",  # 肩垫
    "pauldron",  # 肩甲
    "epaulet",  # 肩章
    "shoulder strap",  # 肩带
    "shoulder accessory",  # 肩部配饰
    # Upper body decorative accessories
    "chest badge",  # 胸前徽章
    "chest logo",  # 胸前标志
    "chest embroidery",  # 胸前刺绣
]

# Prompt words for detecting lower body accessories (belts, waist accessories, etc.)
LOWER_ACCESSORY_PARTS: List[str] = [
    # Waist and fastening system
    "belt",  # 腰带
    "waist belt",  # 腰部腰带
    "tie belt",  # 系带腰带
    "cinch strap",  # 束腰带
    "side strap",  # 侧边系带
    "belt buckle",  # 腰带扣
    "buckle",  # 扣子/搭扣
    "waist tie",  # 腰部系带
    "waist cord",  # 腰绳
    "waist accessory",  # 腰部配饰
    # Lower body bags and pouches
    "pouch",  # 小包/袋
    "side bag",  # 侧包
    "waist bag",  # 腰包
    "hip bag",  # 臀包
    "satchel",  # 挎包
    "bag",  # 包
    # Lower body pockets
    "pocket",  # 口袋
    "flap pocket",  # 有盖口袋
    "zipper pocket",  # 拉链口袋
    "patch pocket",  # 贴袋
]

# Prompt words for detecting exposed flesh-colored skin (will be removed)
# Only detect visible bare skin, not body parts that might be covered by clothing
SKIN_PARTS: List[str] = [
    "human face",  # 人脸
    "facial skin",  # 面部皮肤
    "bare hand",  # 裸露的手
    "bare fingers",  # 裸露的手指
    "bare palm",  # 裸露的手掌
    "bare arm",  # 裸露的手臂
    "bare forearm",  # 裸露的前臂
    "bare upper arm",  # 裸露的上臂
    "bare leg",  # 裸露的腿部
    "bare thigh",  # 裸露的大腿
    "bare calf",  # 裸露的小腿
    "bare foot",  # 裸露的脚
    "bare toes",  # 裸露的脚趾
    "bare neck",  # 裸露的颈部
    "naked skin",  # 裸露的皮肤
]

# Prompt words for detecting background
BACKGROUND_PARTS: List[str] = [
    "background",  # 背景
    "ground",  # 地面
    "floor",  # 地板
    "wall",  # 墙壁
]


def load_models(
    dino_model_name: str,
    device: torch.device,
) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, SAM2ImagePredictor]:
    """
    Load Grounding DINO and SAM 2 models.
    """
    processor = AutoProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name).to(device)
    dino_model.eval()

    from pathlib import Path

    hydra.core.global_hydra.GlobalHydra.instance().clear()

    project_root = Path(__file__).resolve().parent

    hydra.initialize_config_module("sam2_configs", version_base='2.1')

    sam2_checkpoint = project_root / "models" / "sam2.1_hiera_base_plus.pt"
    sam2_config_name = "sam2.1_hiera_b+"

    sam2_model = build_sam2(
        sam2_config_name,
        str(sam2_checkpoint),
        device=device,
    )
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return processor, dino_model, sam2_predictor


def remove_background(
    image_rgb: np.ndarray,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: torch.device,
) -> np.ndarray:
    """
    Step 1: Remove background and return foreground mask.
    """
    print("[STEP 1] 移除背景...")
    image_pil = Image.fromarray(image_rgb)
    
    text = ". ".join(BACKGROUND_PARTS) + "."
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    width, height = image_pil.size
    target_sizes = torch.tensor([[height, width]], device=device)
    
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs["input_ids"],
            box_threshold=0.25, text_threshold=0.2, target_sizes=target_sizes,
        )[0]
    except TypeError:
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"],
                threshold=0.25, target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = processor.post_process_object_detection(
                outputs, threshold=0.25, target_sizes=target_sizes,
            )[0]
    
    bg_boxes = results["boxes"].cpu().numpy()
    
    if bg_boxes.size == 0:
        print("[STEP 1] 未检测到背景，保留整个图像作为前景")
        return np.ones(image_rgb.shape[:2], dtype=bool)
    
    sam2_predictor.set_image(image_rgb)
    bg_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    
    for box in bg_boxes:
        masks, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
        bg_mask = bg_mask | (masks[0] > 0)
    
    foreground_mask = ~bg_mask
    print(f"[STEP 1] 背景移除完成，前景面积: {np.sum(foreground_mask)} 像素")
    return foreground_mask


def remove_skin(
    image_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Remove skin and return (mask after skin removal, skin mask).
    """
    print("[STEP 2] 移除皮肤...")
    image_pil = Image.fromarray(image_rgb)
    
    text = ". ".join(SKIN_PARTS) + "."
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    width, height = image_pil.size
    target_sizes = torch.tensor([[height, width]], device=device)
    
    # Use moderate thresholds to detect bare skin while avoiding clothing
    # The multimask and area check will help filter out clothing
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs["input_ids"],
            box_threshold=0.3, text_threshold=0.25, target_sizes=target_sizes,
        )[0]
    except TypeError:
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"],
                threshold=0.3, target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = processor.post_process_object_detection(
                outputs, threshold=0.3, target_sizes=target_sizes,
            )[0]
    
    skin_boxes = results["boxes"].cpu().numpy()
    labels = results.get("labels", [])
    
    print(f"[STEP 2] 检测到 {len(skin_boxes)} 个皮肤候选区域")
    if len(skin_boxes) > 0 and len(labels) > 0:
        for i, label in enumerate(labels[:5]):  # 只显示前5个
            print(f"  - {i+1}: {label}")
    
    if skin_boxes.size == 0:
        print("[STEP 2] 未检测到皮肤，保留所有前景")
        empty_skin_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        return foreground_mask, empty_skin_mask
    
    sam2_predictor.set_image(image_rgb)
    skin_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    
    # Calculate box area for reference
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    
    for box in skin_boxes:
        # Use multimask to get multiple candidates
        masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=True)
        
        # Calculate area of each mask
        mask_areas = [np.sum(m > 0) for m in masks]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        
        # Select the best mask:
        # 1. Prefer masks that are not too small (at least 30% of box area) to avoid being too conservative
        # 2. Prefer masks that are not too large (not more than 2x box area) to avoid including clothing
        best_idx = None
        best_score = -1
        
        for i, (mask_area, score) in enumerate(zip(mask_areas, scores)):
            coverage = mask_area / box_area if box_area > 0 else 0
            # Prefer masks with coverage between 30% and 200% of box area
            if 0.3 <= coverage <= 2.0:
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        # If no mask meets the criteria, use the one with highest score that's not too large
        if best_idx is None:
            for i, (mask_area, score) in enumerate(zip(mask_areas, scores)):
                coverage = mask_area / box_area if box_area > 0 else 0
                if coverage <= 2.5 and score > best_score:  # Allow slightly larger masks
                    best_score = score
                    best_idx = i
        
        # If still no mask found, use the smallest one (most conservative)
        if best_idx is None:
            best_idx = np.argmin(mask_areas)
        
        best_mask = masks[best_idx] > 0
        mask_area = mask_areas[best_idx]
        
        # Final check: skip masks that are suspiciously large (more than 2.5x box area)
        if mask_area <= box_area * 2.5:
            skin_mask = skin_mask | best_mask
        else:
            print(f"[STEP 2] 跳过可疑的大面积mask (mask面积: {mask_area}, box面积: {box_area}, 覆盖率: {mask_area/box_area:.1f}x)")
    
    # Ensure skin mask only applies within foreground to avoid background being misidentified as skin
    skin_mask = skin_mask & foreground_mask
    
    # Clothing mask = foreground - skin, ensuring result is still within foreground (no background)
    clothing_mask = foreground_mask & (~skin_mask)
    
    print(f"[STEP 2] 皮肤移除完成:")
    print(f"  - 前景面积: {np.sum(foreground_mask)} 像素")
    print(f"  - 皮肤面积: {np.sum(skin_mask)} 像素")
    print(f"  - 剩余面积: {np.sum(clothing_mask)} 像素")
    
    return clothing_mask, skin_mask


def detect_body_part(
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    part_prompts: List[str],
    part_name: str,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: torch.device,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> np.ndarray:
    """
    Generic function to detect a specific body part (headgear, upper garment, lower garment, footwear, gloves).
    Returns mask for the detected part, excluding skin.
    """
    print(f"[检测] 检测{part_name}...")
    image_pil = Image.fromarray(image_rgb)
    
    text = ". ".join(part_prompts) + "."
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    width, height = image_pil.size
    target_sizes = torch.tensor([[height, width]], device=device)
    
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs["input_ids"],
            box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes,
        )[0]
    except TypeError:
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"],
                threshold=box_threshold, target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = processor.post_process_object_detection(
                outputs, threshold=box_threshold, target_sizes=target_sizes,
            )[0]
    
    part_boxes = results["boxes"].cpu().numpy()
    
    if part_boxes.size == 0:
        print(f"[检测] 未检测到{part_name}，返回空 mask")
        return np.zeros(image_rgb.shape[:2], dtype=bool)
    
    sam2_predictor.set_image(image_rgb)
    part_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    
    for box in part_boxes:
        masks, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
        mask = masks[0] > 0
        part_mask = part_mask | mask
    
    # Ensure part mask is within base mask
    part_mask = part_mask & base_mask
    
    print(f"[检测] 检测到{part_name}区域，面积: {np.sum(part_mask)} 像素")
    return part_mask




def save_clothing_with_white_bg(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    output_path: str,
) -> None:
    """
    Save clothing to file with white background.
    """
    if np.sum(mask) == 0:
        print(f"[ERROR] 保存时 mask 为空！无法保存到 {output_path}")
        return
    
    mask_uint8 = (mask.astype(np.uint8)) * 255
    
    white_bg = np.full_like(image_bgr, 255, dtype=np.uint8)
    foreground = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_uint8)
    inv_mask = cv2.bitwise_not(mask_uint8)
    background = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    bgr_out = cv2.add(foreground, background)
    
    cv2.imwrite(output_path, bgr_out)


def save_multi_mask_visualization(
    image_bgr: np.ndarray,
    masks: Dict[str, Tuple[np.ndarray, Tuple[int, int, int]]],
    output_path: str,
    title: str = "",
) -> None:
    """
    Save visualization of multiple masks, each displayed in a different color.
    
    Args:
        image_bgr: Original image
        masks: Dictionary where key is mask name, value is (mask, color) tuple
        output_path: Output path
        title: Title text
    """
    vis = image_bgr.copy()
    
    for name, (mask, color) in masks.items():
        mask_uint8 = (mask.astype(np.uint8)) * 255
        
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
    
    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis)


def save_mask_visualization(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    title: str = "",
    overlay_mask: np.ndarray = None,
    overlay_color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    """
    Save mask visualization (overlay semi-transparent mask on original image).
    
    Args:
        image_bgr: Original image
        mask: Main mask to display
        output_path: Output path
        color: Color of main mask
        title: Title text
        overlay_mask: Optional overlay mask (for displaying removed regions)
        overlay_color: Color of overlay mask
    """
    vis = image_bgr.copy()
    mask_uint8 = (mask.astype(np.uint8)) * 255
    
    if overlay_mask is not None:
        overlay_uint8 = (overlay_mask.astype(np.uint8)) * 255
        overlay_vis = vis.copy()
        overlay_vis[overlay_mask] = overlay_color
        vis = cv2.addWeighted(vis, 0.6, overlay_vis, 0.4, 0)
        contours, _ = cv2.findContours(overlay_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, overlay_color, 2)
    
    overlay = vis.copy()
    overlay[mask] = color
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    
    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imwrite(output_path, vis)


def process_single_image(
    image_path: str,
    output_dir: str,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
    visualize: bool,
) -> None:
    """
    Process single image with body part segmentation:
    1. Remove background
    2. Detect 5 body parts separately: headgear, upper garment, lower garment, footwear, gloves
    3. Save each part as a separate image with white background
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] 无法读取图像: {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Only create debug directory if visualize is enabled
    if visualize:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # Main output directory for final results
    parts_dir = os.path.join(output_dir, "parts_white_bg")
    os.makedirs(parts_dir, exist_ok=True)
    
    print(f"\n[INFO] 开始处理: {os.path.basename(image_path)}")
    
    # Step 1: Remove background
    clothing_base_mask = remove_background(
        image_rgb=image_rgb,
        processor=processor,
        dino_model=dino_model,
        sam2_predictor=sam2_predictor,
        device=device,
    )
    
    if visualize:
        step1_path = os.path.join(debug_dir, f"{base_name}_step1_background_removed.png")
        save_mask_visualization(
            image_bgr=image_bgr,
            mask=clothing_base_mask,
            output_path=step1_path,
            color=(0, 255, 0),
            title="Step 1: Background Removed",
        )
    
    # Step 2: Detect each body part separately
    body_parts = [
        (HEADGEAR_PARTS, "headgear", "帽子"),
        (UPPER_GARMENT_PARTS, "upper_garment", "上衣"),
        (LOWER_GARMENT_PARTS, "lower_garment", "裤子"),
        (FOOTWEAR_PARTS, "footwear", "鞋子"),
        (GLOVES_PARTS, "gloves", "手套"),
    ]
    
    part_masks = {}
    for part_prompts, part_key, part_name in body_parts:
        part_mask = detect_body_part(
            image_rgb=image_rgb,
            base_mask=clothing_base_mask,
            part_prompts=part_prompts,
            part_name=part_name,
            processor=processor,
            dino_model=dino_model,
            sam2_predictor=sam2_predictor,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        part_masks[part_key] = part_mask
    
    # Step 3: Detect accessories separately (upper and lower)
    print("\n[STEP 3] 检测配饰...")
    
    # Detect upper body accessories
    upper_accessory_mask = detect_body_part(
        image_rgb=image_rgb,
        base_mask=clothing_base_mask,
        part_prompts=UPPER_ACCESSORY_PARTS,
        part_name="上衣配饰",
        processor=processor,
        dino_model=dino_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    
    # Detect lower body accessories
    lower_accessory_mask = detect_body_part(
        image_rgb=image_rgb,
        base_mask=clothing_base_mask,
        part_prompts=LOWER_ACCESSORY_PARTS,
        part_name="下衣配饰",
        processor=processor,
        dino_model=dino_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    
    # Step 4: Merge accessories with corresponding body parts and save final results
    print("\n[STEP 4] 合并配饰并保存最终结果...")
    
    # Define output file mapping (simplified names)
    output_files = {
        "hat": ("headgear", "帽子"),
        "coat": ("upper_garment", "上衣"),
        "dress": ("lower_garment", "裤子"),
        "shoes": ("footwear", "鞋子"),
        "gloves": ("gloves", "手套"),
    }
    
    # Merge upper garment + upper accessories
    upper_garment_final = part_masks.get("upper_garment", np.zeros(image_rgb.shape[:2], dtype=bool)) | upper_accessory_mask
    if np.sum(upper_garment_final) > 0:
        kernel = np.ones((3, 3), np.uint8)
        upper_garment_final = cv2.morphologyEx(upper_garment_final.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        output_path = os.path.join(parts_dir, f"{base_name}_coat.png")
        save_clothing_with_white_bg(image_bgr, upper_garment_final, output_path)
        print(f"[INFO] ✓ 上衣: {output_path}")
    
    # Merge lower garment + lower accessories
    lower_garment_final = part_masks.get("lower_garment", np.zeros(image_rgb.shape[:2], dtype=bool)) | lower_accessory_mask
    if np.sum(lower_garment_final) > 0:
        kernel = np.ones((3, 3), np.uint8)
        lower_garment_final = cv2.morphologyEx(lower_garment_final.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        output_path = os.path.join(parts_dir, f"{base_name}_dress.png")
        save_clothing_with_white_bg(image_bgr, lower_garment_final, output_path)
        print(f"[INFO] ✓ 下衣: {output_path}")
    
    # Save other parts
    for output_name, (part_key, part_name) in output_files.items():
        if output_name in ["coat", "dress"]:  # Already saved above
            continue
        part_mask = part_masks.get(part_key, np.zeros(image_rgb.shape[:2], dtype=bool))
        if np.sum(part_mask) > 0:
            output_path = os.path.join(parts_dir, f"{base_name}_{output_name}.png")
            save_clothing_with_white_bg(image_bgr, part_mask, output_path)
            print(f"[INFO] ✓ {part_name}: {output_path}")
    
    # Save combined visualization only if requested
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{base_name}_all.png")
        save_multi_mask_visualization(
            image_bgr=image_bgr,
            masks={
                "Headgear": (part_masks.get("headgear", np.zeros(image_rgb.shape[:2], dtype=bool)), (0, 255, 255)),
                "Upper Garment": (upper_garment_final, (0, 255, 0)),
                "Lower Garment": (lower_garment_final, (255, 255, 0)),
                "Footwear": (part_masks.get("footwear", np.zeros(image_rgb.shape[:2], dtype=bool)), (255, 0, 255)),
                "Gloves": (part_masks.get("gloves", np.zeros(image_rgb.shape[:2], dtype=bool)), (255, 0, 0)),
            },
            output_path=vis_path,
            title="All Parts",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Automatic FFXIV gear extraction using Grounding DINO + SAM 2",
    )
    parser.add_argument(
        "--dino-model-name",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO model name on HuggingFace",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO box confidence threshold",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text matching threshold",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization (no bounding boxes or segmentation edges on original image)",
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, "images")
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    print("[INFO] 正在加载 Grounding DINO 与 SAM 2 模型...")
    processor, dino_model, sam2_predictor = load_models(
        dino_model_name=args.dino_model_name,
        device=device,
    )
    print("[INFO] 模型加载完成。")

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ]
    image_files.sort()

    if not image_files:
        print(f"[WARN] 在目录中未找到图片文件: {input_dir}")
        return

    print(f"[INFO] 共找到 {len(image_files)} 张图片，开始批处理...")

    for idx, fname in enumerate(image_files, start=1):
        img_path = os.path.join(input_dir, fname)
        print(f"[INFO] ({idx}/{len(image_files)}) 处理 {fname} ...")
        process_single_image(
            image_path=img_path,
            output_dir=output_dir,
            processor=processor,
            dino_model=dino_model,
            sam2_predictor=sam2_predictor,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            visualize=not args.no_visualize,
        )

    print("[INFO] 全部处理完成。")


if __name__ == "__main__":
    main()


