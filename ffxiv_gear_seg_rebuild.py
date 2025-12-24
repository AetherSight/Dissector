import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -----------------------------
# Model paths (fixed)
# -----------------------------
# Use absolute paths based on project root for reliability
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM2_CONFIG_NAME = "sam2.1_hiera_b+"
SAM2_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "sam2.1_hiera_base_plus.pt")


# -----------------------------
# Prompts
# -----------------------------
HEADWEAR_PROMPTS: List[str] = [
    # head / face region (for removal)
    "head",
    "human head",
    "face",
    "facial area",
    "hair",
    "hairstyle",
    "ponytail hair",
    "cat ear",
    "animal ear",
    # headwear (fallbacks if present)
    "headwear",
    "hat",
    "cap",
    "helmet",
    "crown",
    "tiara",
    "headband",
    "hood",
]

UPPER_PROMPTS: List[str] = [
    # main garment forms
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
    # dresses (treat as upper to keep bodice/torso)
    "dress bodice",
    "dress top",
    "upper part of dress",
    # sleeves / arms / decorations
    "sleeve",
    "long sleeve",
    "short sleeve",
    "arm guard",
    "bracer",
    "arm band",
    "arm accessory",
    # lining / fabric body
    "garment body",
    "clothing fabric",
    "inner lining",
]

LOWER_PROMPTS: List[str] = [
    # pants-focused to avoid full body
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

SKIN_PROMPTS_FACE_NECK: List[str] = []
SKIN_PROMPTS_LIMBS: List[str] = []


# -----------------------------
# Utility
# -----------------------------
def load_models(dino_model_name: str, device: torch.device) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, SAM2ImagePredictor]:
    processor = AutoProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name).to(device)

    # Follow the loading pattern used in ffxiv_gear_seg_pipeline.py (Hydra config name)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("sam2_configs", version_base="2.1")

    sam2_model = build_sam2(
        SAM2_CONFIG_NAME,
        SAM2_CHECKPOINT_PATH,
        device=device,
    )
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return processor, dino_model, sam2_predictor


def run_grounding_dino(
    image_pil: Image.Image,
    prompts: List[str],
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
):
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
            box_threshold=box_threshold,
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


def mask_from_boxes(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    sam2_predictor: SAM2ImagePredictor,
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(image_rgb.shape[:2], dtype=bool)

    sam2_predictor.set_image(image_rgb)
    mask_total = np.zeros(image_rgb.shape[:2], dtype=bool)
    for box in boxes:
        masks, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
        mask = masks[0] > 0
        mask_total |= mask

    # small closing to fill pinholes
    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # remove small speckles
    mask_total = remove_small_components(mask_total, min_area_ratio=min_area_ratio)
    return mask_total


def remove_small_components(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """Remove tiny speckles using connected components."""
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


def save_with_white_bg(image_bgr: np.ndarray, mask: np.ndarray, output_path: str):
    if np.sum(mask) == 0:
        print(f"[WARN] mask is empty, skip save: {output_path}")
        return
    mask_uint8 = (mask.astype(np.uint8)) * 255
    white = np.full_like(image_bgr, 255, dtype=np.uint8)
    fg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_uint8)
    bg = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask_uint8))
    out = cv2.add(fg, bg)
    cv2.imwrite(output_path, out)
    print(f"[INFO] saved: {output_path}")


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


def process_image(
    image_path: str,
    output_dir: str,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] cannot read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # per-image output dirs
    img_out_dir = os.path.join(output_dir, base)
    debug_dir = os.path.join(output_dir, "debug", base)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # Detect in a fixed order so we can refine masks with each other
    masks = {}

    def detect_and_store(key: str, prompts: List[str]):
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
        masks[key] = mask_from_boxes(image_rgb, boxes, sam2_predictor)

    # 1) shoes first (used to clean lower)
    print("[STEP] detecting shoes ...")
    detect_and_store("shoes", FOOTWEAR_PROMPTS)
    save_debug_overlay(image_bgr, masks["shoes"], os.path.join(debug_dir, "step1_shoes.jpg"), (255, 0, 255), "shoes")

    # 2) lower
    print("[STEP] detecting lower ...")
    detect_and_store("lower_raw", LOWER_PROMPTS)
    lower_mask = masks.get("lower_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    masks["lower"] = lower_mask
    save_debug_overlay(image_bgr, lower_mask, os.path.join(debug_dir, "step2_lower.jpg"), (255, 255, 0), "lower")

    # 3) head (only for removal, no saving)
    print("[STEP] detecting head (for removal) ...")
    detect_and_store("head", HEADWEAR_PROMPTS)
    # Expand head mask to ensure removal of head region (no arbitrary top strip)
    head_mask = masks.get("head", np.zeros(image_rgb.shape[:2], dtype=bool))
    if np.any(head_mask):
        kernel = np.ones((15, 15), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["head"] = head_mask
    save_debug_overlay(image_bgr, head_mask, os.path.join(debug_dir, "step3_head.jpg"), (0, 165, 255), "head (remove)")

    # 4) upper
    print("[STEP] detecting upper ...")
    detect_and_store("upper_raw", UPPER_PROMPTS)
    upper_mask = masks.get("upper_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    # remove lower, shoes, head spill
    upper_mask = (
        upper_mask
        & (~masks.get("lower", np.zeros_like(upper_mask)))
        & (~masks.get("shoes", np.zeros_like(upper_mask)))
        & (~masks.get("head", np.zeros_like(upper_mask)))
    )
    upper_mask = remove_small_components(upper_mask, min_area_ratio=0.001)
    masks["upper"] = upper_mask
    masks["lower"] = remove_small_components(masks.get("lower", np.zeros_like(upper_mask)), min_area_ratio=0.001)
    masks["shoes"] = remove_small_components(masks.get("shoes", np.zeros_like(upper_mask)), min_area_ratio=0.001)
    save_debug_overlay(image_bgr, upper_mask, os.path.join(debug_dir, "step4_upper.jpg"), (0, 255, 0), "upper")

    # Skip skin removal (per request)

    # Save final outputs
    outputs = [
        ("upper", "upper"),
        ("lower", "lower"),
        ("shoes", "shoes"),
        ("head", "head"),  # optional head output (can be ignored)
    ]
    for key, name in outputs:
        out_path = os.path.join(img_out_dir, f"{name}.jpg")
        save_with_white_bg(image_bgr, masks.get(key, np.zeros(image_rgb.shape[:2], dtype=bool)), out_path)


def main():
    parser = argparse.ArgumentParser(description="FFXIV gear segmentation (head / upper / lower / shoes)")
    parser.add_argument("--dino-model-name", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, "images")
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print("[INFO] loading models ...")
    processor, dino_model, sam2_predictor = load_models(args.dino_model_name, device)
    print("[INFO] models loaded.")

    images = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"[INFO] found {len(images)} images.")

    for img in images:
        print(f"\n[INFO] processing {os.path.basename(img)}")
        process_image(
            image_path=img,
            output_dir=output_dir,
            processor=processor,
            dino_model=dino_model,
            sam2_predictor=sam2_predictor,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

    print("\n[INFO] done.")


if __name__ == "__main__":
    main()

