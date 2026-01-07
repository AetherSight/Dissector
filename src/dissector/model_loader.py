"""
Model loader for different platforms.
Mac: MLX SAM3
Windows/Linux: Facebook SAM3 (CUDA)
"""
import os
import platform
import logging
from typing import Tuple, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

SAM3Model = Any


def is_mac() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def load_sam3_model() -> Tuple[SAM3Model, str]:
    """
    Load SAM3 model based on platform.
    
    Returns:
        Tuple of (model, platform_type) where platform_type is "mlx" or "cuda"
    """
    if is_mac():
        return _load_mlx_sam3(), "mlx"
    else:
        return _load_cuda_sam3(), "cuda"


def _load_mlx_sam3() -> SAM3Model:
    """
    Load MLX SAM3 model for Mac.
    
    Note: MLX SAM3 requires the mlx-sam3 package.
    Install with: pip install mlx-sam3 or from GitHub:
    git clone https://github.com/Deekshith-Dade/mlx_sam3.git
    cd mlx_sam3 && pip install -e .
    """
    try:
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            try:
                from mlx_sam3 import build_sam3_image_model
                from mlx_sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                raise ImportError("MLX SAM3 not found. Install with: pip install mlx-sam3")
        
        logger.info("Loading MLX SAM3 model...")
        model = build_sam3_image_model()
        processor = Sam3Processor(model, confidence_threshold=0.5)
        
        class MLXSam3Wrapper:
            def __init__(self, model, processor):
                self.model = model
                self.processor = processor
                self._current_state = None
                self._current_image = None  # Store image id instead of image object
            
            def generate_mask_from_bbox(self, image_pil: Image.Image, bbox: np.ndarray) -> np.ndarray:
                """
                Generate mask from bounding box using MLX SAM3.
                
                Args:
                    image_pil: PIL Image
                    bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2]
                
                Returns:
                    Binary mask as numpy array
                """
                h, w = image_pil.size[1], image_pil.size[0]
                
                image_id = id(image_pil)
                if self._current_image != image_id:
                    self._current_state = self.processor.set_image(image_pil)
                    self._current_image = image_id
                
                x1, y1, x2, y2 = bbox[0]
                box_normalized = np.array([[x1 / w, y1 / h, x2 / w, y2 / h]], dtype=np.float32)
                
                try:
                    if hasattr(self.processor, 'set_box_prompt'):
                        state = self.processor.set_box_prompt(box_normalized, self._current_state)
                    elif hasattr(self.processor, 'set_bbox_prompt'):
                        state = self.processor.set_bbox_prompt(box_normalized, self._current_state)
                    else:
                        state = self._current_state.copy() if isinstance(self._current_state, dict) else self._current_state
                        if isinstance(state, dict):
                            state['boxes'] = box_normalized
                        else:
                            state = self.processor(box_normalized, self._current_state)
                except Exception as e:
                    logger.warning(f"Failed to set box prompt with primary method: {e}, trying alternative")
                    try:
                        state = self.processor(box_normalized, self._current_state)
                    except:
                        logger.error(f"Failed to generate mask from bbox: {e}")
                        return np.zeros((h, w), dtype=bool)
                
                if isinstance(state, dict):
                    masks = state.get("masks", [])
                else:
                    masks = getattr(state, "masks", [])
                
                if not masks or len(masks) == 0:
                    return np.zeros((h, w), dtype=bool)
                
                mask = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
                
                if mask.dtype != bool:
                    mask = mask > 0.5 if mask.max() <= 1.0 else mask > 127
                
                if mask.shape != (h, w):
                    from PIL import Image as PILImage
                    mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
                    mask_pil = PILImage.fromarray(mask_uint8)
                    mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
                    mask = np.array(mask_pil) > 127
                
                return mask.astype(bool)
        
        return MLXSam3Wrapper(model, processor)
    except ImportError as e:
        logger.error(f"Failed to import MLX SAM3: {e}")
        raise RuntimeError("MLX SAM3 is required on macOS. Install with: pip install mlx-sam3") from e


def _load_cuda_sam3() -> SAM3Model:
    """Load Facebook SAM3 model for Windows/Linux (CUDA)."""
    from sam3.model_builder import build_sam3_image_model
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    bpe_path = os.path.join(models_dir, "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        logger.warning(f"BPE file not found at {bpe_path}, using default path")
        bpe_path = None
    
    checkpoint_path = os.path.join(models_dir, "sam3.pt")
    if not os.path.exists(checkpoint_path):
        logger.info(f"Model checkpoint not found at {checkpoint_path}, Facebook SAM3 will download it")
        checkpoint_path = None
    if bpe_path and checkpoint_path:
        logger.info(f"Loading Facebook SAM3 model from: {checkpoint_path}")
        logger.info(f"Using BPE file from: {bpe_path}")
        return build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
    elif bpe_path:
        logger.info(f"Using BPE file from: {bpe_path}")
        return build_sam3_image_model(bpe_path=bpe_path)
    elif checkpoint_path:
        logger.info(f"Loading model from: {checkpoint_path}")
        return build_sam3_image_model(checkpoint_path=checkpoint_path)
    else:
        logger.info("Using default paths (Facebook SAM3 will download if needed)")
        return build_sam3_image_model()

