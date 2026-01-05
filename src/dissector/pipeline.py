import os
import base64
import logging
import platform
import time
from typing import Dict, List, Tuple, Optional

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
from .segmentation import segment_parts


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    Priority: CUDA > CPU
    On macOS with MLX backend, use CPU to avoid concurrency issues.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_models(
    device: torch.device,
    sam3_backend: Optional[str] = None,
) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, SAM3Base]:
    device_str = "cuda" if device.type == "cuda" else "cpu"
    sam3_model = SAM3Factory.create(backend=sam3_backend, device=device_str)
    logger.info(f"Loaded SAM3 model with backend: {sam3_model.backend_name}")

    if sam3_model.backend_name == "mlx":
        logger.info("MLX backend detected: skipping Grounding DINO loading.")
        processor = None
        dino_model = None
        return processor, dino_model, sam3_model

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dino_local_dir = os.path.join(project_root, "models", "grounding-dino-base")

    logger.info(f"Loading Grounding DINO from local path: {dino_local_dir}")

    processor = AutoProcessor.from_pretrained(
        dino_local_dir,
        local_files_only=True,
    )
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_local_dir,
        local_files_only=True,
    ).to(device)

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

    # 统一使用 segment_parts，它会根据后端类型自动选择实现
    logger.info(f"[PERF] {sam3_model.backend_name.upper()} backend: using segment_parts")
    segment_start = time.time()
    results = segment_parts(
        image_pil=image_pil,
        sam3_model=sam3_model,
        processor=processor,
        dino_model=dino_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    segment_time = time.time() - segment_start
    logger.info(f"[PERF] segment_parts: {segment_time:.2f}s")
    return results


def run_batch(
    input_dir: str,
    output_dir: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> List[Dict[str, str]]:
    device = get_device()
    logger.info(f"device: {device} (platform: {platform.system()})")
    logger.info("loading models ...")
    processor, dino_model, sam3_model = load_models(device)
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

