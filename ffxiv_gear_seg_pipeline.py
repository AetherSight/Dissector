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


# Prompt words for detecting clothing and accessories
CLOTHING_PARTS: List[str] = [
    "chest armor",
    "armor",
    "arm band",
    "armlet",
    "bracer",
    "wristband",
    "shoulder pad",
    "pauldron",
    "clothing",
    "garment",
]

# Prompt words for detecting skin/body parts (will be removed)
SKIN_PARTS: List[str] = [
    "skin",
    "face",
    "hand",
    "arm",
    "leg",
    "body",
    "head",
    "neck",
]

# Prompt words for detecting background
BACKGROUND_PARTS: List[str] = [
    "background",
    "ground",
    "floor",
    "wall",
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
    
    skin_boxes = results["boxes"].cpu().numpy()
    
    if skin_boxes.size == 0:
        print("[STEP 2] 未检测到皮肤，保留所有前景")
        empty_skin_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        return foreground_mask, empty_skin_mask
    
    sam2_predictor.set_image(image_rgb)
    skin_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    
    for box in skin_boxes:
        masks, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
        skin_mask = skin_mask | (masks[0] > 0)
    
    # Ensure skin mask only applies within foreground to avoid background being misidentified as skin
    skin_mask = skin_mask & foreground_mask
    
    # Clothing mask = foreground - skin, ensuring result is still within foreground (no background)
    clothing_mask = foreground_mask & (~skin_mask)
    
    print(f"[STEP 2] 皮肤移除完成:")
    print(f"  - 前景面积: {np.sum(foreground_mask)} 像素")
    print(f"  - 皮肤面积: {np.sum(skin_mask)} 像素")
    print(f"  - 剩余面积: {np.sum(clothing_mask)} 像素")
    
    return clothing_mask, skin_mask


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
    Process single image with three-step pipeline:
    1. Remove background
    2. Remove skin
    3. Segment and merge clothing (including small accessories)
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] 无法读取图像: {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"\n[INFO] 开始处理: {os.path.basename(image_path)}")
    
    foreground_mask = remove_background(
        image_rgb=image_rgb,
        processor=processor,
        dino_model=dino_model,
        sam2_predictor=sam2_predictor,
        device=device,
    )
    
    step1_path = os.path.join(debug_dir, f"{base_name}_step1_foreground.png")
    save_mask_visualization(
        image_bgr=image_bgr,
        mask=foreground_mask,
        output_path=step1_path,
        color=(0, 255, 0),
        title="Step 1: Foreground (Background Removed)",
    )
    print(f"[DEBUG] 已保存步骤1结果到: {step1_path}")
    
    clothing_base_mask, skin_mask = remove_skin(
        image_rgb=image_rgb,
        foreground_mask=foreground_mask,
        processor=processor,
        dino_model=dino_model,
        sam2_predictor=sam2_predictor,
        device=device,
    )
    
    # Detect clothing regions for step 2 visualization and step 3 processing
    image_pil = Image.fromarray(image_rgb)
    text = ". ".join(CLOTHING_PARTS) + "."
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    width, height = image_pil.size
    target_sizes = torch.tensor([[height, width]], device=device)
    
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
    
    clothing_boxes_debug = results["boxes"].cpu().numpy()
    
    clothing_mask_debug = np.zeros(image_rgb.shape[:2], dtype=bool)
    if clothing_boxes_debug.size > 0:
        sam2_predictor.set_image(image_rgb)
        for box in clothing_boxes_debug:
            masks, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
            clothing_mask_debug = clothing_mask_debug | (masks[0] > 0)
        clothing_mask_debug = clothing_mask_debug & foreground_mask
    
    step2_path = os.path.join(debug_dir, f"{base_name}_step2_no_skin.png")
    save_multi_mask_visualization(
        image_bgr=image_bgr,
        masks={
            "Skin (Red)": (skin_mask, (0, 0, 255)),
            "Clothing (Yellow)": (clothing_mask_debug, (0, 255, 255)),
            "Remaining (Green)": (clothing_base_mask, (0, 255, 0)),
        },
        output_path=step2_path,
        title="Step 2: Red=Skin, Yellow=Clothing, Green=Remaining",
    )
    print(f"[DEBUG] 已保存步骤2结果到: {step2_path}")
    
    print("[STEP 3] 合并衣服区域和褶皱...")
    
    if np.sum(clothing_mask_debug) > 0:
        # Strategy: Use detected clothing (yellow) as base, then supplement with base mask (green) for wrinkles
        # 1. Dilate yellow mask to include wrinkle regions
        kernel_expand = np.ones((10, 10), np.uint8)
        expanded_yellow = cv2.dilate(clothing_mask_debug.astype(np.uint8), kernel_expand, iterations=1).astype(bool)
        
        # 2. Use green mask within expanded yellow region to supplement wrinkles
        final_clothing_mask = (clothing_base_mask & expanded_yellow) | clothing_mask_debug
    else:
        final_clothing_mask = clothing_base_mask
    
    # Morphological operations: fill small holes (use small kernel to preserve wrinkles)
    kernel = np.ones((3, 3), np.uint8)
    final_clothing_mask = cv2.morphologyEx(final_clothing_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    
    print(f"[STEP 3] 最终面积: {np.sum(final_clothing_mask)} 像素")
    
    step3_path = os.path.join(debug_dir, f"{base_name}_step3_clothing.png")
    save_multi_mask_visualization(
        image_bgr=image_bgr,
        masks={
            "Base Mask (Green)": (clothing_base_mask, (0, 255, 0)),
            "Detected Clothing (Yellow)": (clothing_mask_debug, (0, 255, 255)),
            "Final Result (Cyan)": (final_clothing_mask, (255, 255, 0)),
        },
        output_path=step3_path,
        title="Step 3: Green=Base, Yellow=Detected, Cyan=Final",
    )
    print(f"[DEBUG] 已保存步骤3结果到: {step3_path}")
    
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    output_path = os.path.join(crops_dir, f"{base_name}_clothing.png")
    
    if np.sum(final_clothing_mask) == 0:
        print("[ERROR] 最终 mask 为空，无法保存！")
        return
    
    save_clothing_with_white_bg(image_bgr, final_clothing_mask, output_path)
    print(f"[INFO] 已保存最终衣服到: {output_path}")
    
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        vis_path = os.path.join(vis_dir, f"{base_name}_vis.png")
        save_mask_visualization(
            image_bgr=image_bgr,
            mask=final_clothing_mask,
            output_path=vis_path,
            color=(0, 255, 0),
        )
        print(f"[INFO] 已保存可视化到: {vis_path}")


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


