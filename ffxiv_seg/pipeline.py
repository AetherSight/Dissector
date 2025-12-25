import os
import shutil
from typing import Dict, List, Tuple

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_CONFIG_NAME = "sam2.1_hiera_b+"
SAM2_CONFIG_DIR = os.path.join(PROJECT_ROOT, "models", "sam2_configs")
SAM2_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "sam2.1_hiera_base_plus.pt")

# -----------------------------
# Prompts
# -----------------------------
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

HAND_PROMPTS: List[str] = [
    "human hand",
    "hands",
    "palm",
    "fingers",
    "bare hand",
    "bare fingers",
]


# -----------------------------
# Model loading
# -----------------------------
def load_models(dino_model_name: str, device: torch.device) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, SAM2ImagePredictor]:
    processor = AutoProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name).to(device)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=SAM2_CONFIG_DIR, version_base="2.1")

    sam2_model = build_sam2(
        SAM2_CONFIG_NAME,
        SAM2_CHECKPOINT_PATH,
        device=device,
    )
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return processor, dino_model, sam2_predictor


# -----------------------------
# Helpers
# -----------------------------
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

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    mask_total = remove_small_components(mask_total, min_area_ratio=min_area_ratio)
    return mask_total


def save_with_white_bg(image_bgr: np.ndarray, mask: np.ndarray, output_path: str):
    if np.sum(mask) == 0:
        print(f"[WARN] mask is empty, skip save: {output_path}")
        return
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = 5
    y0 = max(0, y0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)

    mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    img_crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]

    h, w = mask_crop.shape
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    mask_canvas = np.zeros((side, side), dtype=np.uint8)

    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img_crop
    mask_canvas[y_off : y_off + h, x_off : x_off + w] = (mask_crop.astype(np.uint8)) * 255

    fg = cv2.bitwise_and(canvas, canvas, mask=mask_canvas)
    bg = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask_canvas))
    out = cv2.add(fg, bg)

    # resize to 512x512 for consistent centered output
    out = cv2.resize(out, (512, 512), interpolation=cv2.INTER_AREA)
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


# -----------------------------
# Main processing
# -----------------------------
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

    img_out_dir = os.path.join(output_dir, base)
    debug_dir = os.path.join(output_dir, "debug", base)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    masks: Dict[str, np.ndarray] = {}

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

    # 1) shoes
    print("[STEP] detecting shoes ...")
    detect_and_store("shoes", FOOTWEAR_PROMPTS)
    save_debug_overlay(image_bgr, masks["shoes"], os.path.join(debug_dir, "step1_shoes.jpg"), (255, 0, 255), "shoes")

    # 2) lower
    print("[STEP] detecting lower ...")
    detect_and_store("lower_raw", LOWER_PROMPTS)
    lower_mask = masks.get("lower_raw", np.zeros(image_rgb.shape[:2], dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    masks["lower"] = remove_small_components(lower_mask, min_area_ratio=0.001)
    save_debug_overlay(image_bgr, masks["lower"], os.path.join(debug_dir, "step2_lower.jpg"), (255, 255, 0), "lower")

    # 3) head (remove only)
    print("[STEP] detecting head (for removal) ...")
    detect_and_store("head", HEADWEAR_PROMPTS)
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
    upper_mask = (
        upper_mask
        & (~masks.get("lower", np.zeros_like(upper_mask)))
        & (~masks.get("shoes", np.zeros_like(upper_mask)))
        & (~masks.get("head", np.zeros_like(upper_mask)))
    )
    upper_mask = remove_small_components(upper_mask, min_area_ratio=0.001)
    masks["upper"] = upper_mask
    masks["shoes"] = remove_small_components(masks.get("shoes", np.zeros_like(upper_mask)), min_area_ratio=0.001)
    save_debug_overlay(image_bgr, upper_mask, os.path.join(debug_dir, "step4_upper.jpg"), (0, 255, 0), "upper")

    # 5) hands removal from upper
    print("[STEP] detecting hands (remove from upper)...")
    detect_and_store("hands", HAND_PROMPTS)
    hand_mask = masks.get("hands", np.zeros(image_rgb.shape[:2], dtype=bool))
    hand_mask = remove_small_components(hand_mask, min_area_ratio=0.0005)
    if np.any(hand_mask):
        kernel = np.ones((5, 5), np.uint8)
        hand_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["hands"] = hand_mask
    save_debug_overlay(image_bgr, hand_mask, os.path.join(debug_dir, "step5_hands.jpg"), (0, 0, 255), "hands (remove)")

    masks["upper"] = masks.get("upper", np.zeros_like(hand_mask)) & (~hand_mask)
    masks["upper"] = remove_small_components(masks["upper"], min_area_ratio=0.001)
    save_debug_overlay(image_bgr, masks["upper"], os.path.join(debug_dir, "step6_upper_nohands.jpg"), (0, 200, 0), "upper - hands")

    # Save final outputs
    outputs = [
        ("upper", "upper"),
        ("lower", "lower"),
        ("shoes", "shoes"),
        ("head", "head"),
        ("hands", "hands"),
    ]
    for key, name in outputs:
        out_path = os.path.join(img_out_dir, f"{name}.jpg")
        save_with_white_bg(image_bgr, masks.get(key, np.zeros(image_rgb.shape[:2], dtype=bool)), out_path)


def run_batch(
    input_dir: str,
    output_dir: str,
    dino_model_name: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
):
    # clean outputs each run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print("[INFO] loading models ...")
    processor, dino_model, sam2_predictor = load_models(dino_model_name, device)
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
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    print("\n[INFO] done.")


