import os
import base64
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
logging.getLogger("transformers.image_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.WARNING)

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BPE_PATH = os.path.join(PROJECT_ROOT, "assets", "bpe_simple_vocab_16e6.txt.gz")
SAM3_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "sam3.pt")

def load_models(dino_model_name: str, device: torch.device) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, Sam3Processor]:
    processor = AutoProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name).to(device)

    sam3_model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        device=device,
        checkpoint_path=SAM3_CHECKPOINT_PATH,
        load_from_HF=False,
    )
    sam3_model = sam3_model.to(device)
    sam3_processor = Sam3Processor(sam3_model)
    return processor, dino_model, sam3_processor


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
    image_pil: Image.Image,
    boxes: np.ndarray,
    sam3_processor: Sam3Processor,
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    if boxes.size == 0:
        h, w = image_pil.size[1], image_pil.size[0]
        return np.zeros((h, w), dtype=bool)

    inference_state = sam3_processor.set_image(image_pil)
    mask_total = None
    h, w = image_pil.size[1], image_pil.size[0]

    for box in boxes:
        sam3_processor.reset_all_prompts(inference_state)

        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0 / w
        center_y = (y1 + y2) / 2.0 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        box_normalized = [center_x, center_y, width, height]

        output = sam3_processor.add_geometric_prompt(
            box=box_normalized,
            label=True,
            state=inference_state
        )

        masks = output.get("masks", None)
        if masks is not None and masks.numel() > 0:
            masks_np = masks.cpu().numpy()

            if masks_np.ndim == 4:
                masks_2d = masks_np.squeeze(1)
                mask = np.any(masks_2d, axis=0).astype(bool)
            elif masks_np.ndim == 3:
                mask = np.any(masks_np, axis=0).astype(bool)
            elif masks_np.ndim == 2:
                mask = masks_np.astype(bool)
            else:
                mask = masks_np.squeeze()
                if mask.ndim == 3:
                    mask = np.any(mask, axis=0).astype(bool)
                elif mask.ndim == 2:
                    mask = mask.astype(bool)
                else:
                    logger.warning(f"Unexpected mask shape after squeeze: {mask.shape}, skipping...")
                    continue

            if mask.ndim != 2:
                logger.warning(f"Mask is not 2D after processing: {mask.shape}, skipping...")
                continue

            if mask.shape != (h, w):
                logger.warning(f"Mask shape {mask.shape} doesn't match image size ({h}, {w}), resizing...")
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

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


def process_image(
    image_path: str,
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    sam3_processor: Sam3Processor,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, str]:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] cannot read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

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
        masks[key] = mask_from_boxes(image_pil, boxes, sam3_processor)

    logger.debug("[STEP] detecting shoes ...")
    detect_and_store("shoes", FOOTWEAR_PROMPTS)

    logger.debug("[STEP] detecting lower ...")
    detect_and_store("lower_raw", LOWER_PROMPTS)
    lower_mask = masks.get("lower_raw", np.zeros((h, w), dtype=bool))
    lower_mask = lower_mask & (~masks.get("shoes", np.zeros_like(lower_mask)))
    masks["lower"] = remove_small_components(lower_mask, min_area_ratio=0.001)

    logger.debug("[STEP] detecting head (for removal) ...")
    detect_and_store("head", HEADWEAR_PROMPTS)
    head_mask = masks.get("head", np.zeros((h, w), dtype=bool))
    if np.any(head_mask):
        kernel = np.ones((15, 15), np.uint8)
        head_mask = cv2.dilate(head_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    masks["head"] = head_mask

    logger.debug("[STEP] detecting upper ...")
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

    logger.debug("[STEP] detecting hands (remove from upper)...")
    detect_and_store("hands", HAND_PROMPTS)
    hand_mask = masks.get("hands", np.zeros(image_rgb.shape[:2], dtype=bool))
    hand_mask = remove_small_components(hand_mask, min_area_ratio=0.0005)
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
    for key, name in outputs:
        mask_part = masks.get(key, np.zeros((h, w), dtype=bool))
        out_img = render_white_bg(image_bgr, mask_part)
        results[name] = encode_bgr_to_base64(out_img, ext=".jpg")
    return results


def run_batch(
    input_dir: str,
    output_dir: str,
    dino_model_name: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> List[Dict[str, str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info("loading models ...")
    processor, dino_model, sam3_processor = load_models(dino_model_name, device)
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
            sam3_processor=sam3_processor,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        results_all.append(res)
        logger.info(f"done {os.path.basename(img)}")

    return results_all


