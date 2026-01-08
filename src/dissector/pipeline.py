import os
import base64
import logging
import time
from typing import Dict, Tuple, Optional, Union, Any

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
import io

from .backend import SAM3Factory, SAM3Base
from .segmentation import segment_parts


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_models(
    device: torch.device,
    sam3_backend: Optional[str] = None,
) -> Tuple[None, None, SAM3Base]:
    device_str = "cuda" if device.type == "cuda" else "cpu"
    sam3_model = SAM3Factory.create(backend=sam3_backend, device=device_str)
    logger.info(f"Loaded SAM3 model with backend: {sam3_model.backend_name}")

    return None, None, sam3_model


def prepare_image_for_backend(
    image_pil: Image.Image,
    sam3_model: SAM3Base,
) -> Image.Image:
    buffer = io.BytesIO()
    if sam3_model.backend_name == "mlx":
        image_pil.save(buffer, format="JPEG", quality=100)
    else:
        image_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def remove_background(
    image: Union[str, Image.Image],
    processor: Optional[Any],
    dino_model: Optional[Any],
    sam3_model: SAM3Base,
    device: torch.device,
) -> str:
    if isinstance(image, str):
        image_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Cannot read image: {image}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
    else:
        image_pil = image
    
    image_pil = prepare_image_for_backend(image_pil, sam3_model)
    image_rgb = np.array(image_pil)
    h, w = image_rgb.shape[:2]
    
    person_prompts = [
        "person",
        "human",
    ]
    
    logger.info("Using SAM3 text prompt for background removal")
    person_mask_total = None
    for prompt in person_prompts:
        try:
            prompt_mask = sam3_model.generate_mask_from_text_prompt(
                image_pil=image_pil,
                text_prompt=prompt,
            )
            if prompt_mask is not None and prompt_mask.size > 0 and np.sum(prompt_mask) > 0:
                if prompt_mask.shape != (h, w):
                    mask_uint8 = (prompt_mask.astype(np.uint8) * 255) if prompt_mask.dtype == bool else prompt_mask.astype(np.uint8)
                    prompt_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    if prompt_mask.dtype != bool:
                        prompt_mask = prompt_mask.astype(bool)
                
                if person_mask_total is None:
                    person_mask_total = prompt_mask.copy()
                else:
                    person_mask_total |= prompt_mask
        except Exception as e:
            logger.warning(f"Failed to generate mask for prompt '{prompt}': {e}")
            continue
    
    if person_mask_total is None or np.sum(person_mask_total) == 0:
        person_mask = np.ones((h, w), dtype=bool)
    else:
        person_mask = person_mask_total
    
    image_rgba = image_rgb.copy()
    alpha_channel = (person_mask.astype(np.uint8) * 255)
    image_rgba = np.dstack([image_rgba, alpha_channel])
    
    image_pil_rgba = Image.fromarray(image_rgba, "RGBA")
    
    buffer = io.BytesIO()
    image_pil_rgba.save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_image(
    image: Union[str, Image.Image],
    processor: Optional[Any],
    dino_model: Optional[Any],
    sam3_model: SAM3Base,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, str]:
    if isinstance(image, str):
        image_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.warning(f"Cannot read image: {image}")
            return {}
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
    else:
        image_pil = image
    
    image_pil = prepare_image_for_backend(image_pil, sam3_model)
    image_rgb = np.array(image_pil)
    h, w = image_rgb.shape[:2]

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

