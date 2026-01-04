"""
Model loader for different platforms.
Mac: MLX SAM3
Windows/Linux: Ultralytics SAM
"""
import os
import platform
import logging
from typing import Union, Tuple, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Type hints for model objects
SAM3Model = Any

def is_mac() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"

def load_sam3_model() -> Tuple[SAM3Model, str]:
    """
    Load SAM3 model based on platform.
    
    Returns:
        Tuple of (model, platform_type) where platform_type is "mlx" or "ultralytics"
    """
    if is_mac():
        return _load_mlx_sam3(), "mlx"
    else:
        return _load_ultralytics_sam3(), "ultralytics"

def _load_mlx_sam3() -> SAM3Model:
    """
    Load MLX SAM3 model for Mac.
    
    Note: MLX SAM3 requires the mlx-sam3 package.
    Install with: pip install mlx-sam3 or from GitHub:
    git clone https://github.com/Deekshith-Dade/mlx_sam3.git
    cd mlx_sam3 && pip install -e .
    """
    try:
        # Try importing MLX SAM3
        # The package name might be 'sam3' (MLX version) or 'mlx_sam3'
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            # Try alternative import path
            try:
                from mlx_sam3 import build_sam3_image_model
                from mlx_sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                raise ImportError("MLX SAM3 not found. Install with: pip install mlx-sam3")
        
        logger.info("Loading MLX SAM3 model...")
        model = build_sam3_image_model()
        processor = Sam3Processor(model, confidence_threshold=0.5)
        
        # Wrap model and processor together
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
                
                # Check if we need to set a new image
                # Use image id for comparison to avoid storing full image
                image_id = id(image_pil)
                if self._current_image != image_id:
                    self._current_state = self.processor.set_image(image_pil)
                    self._current_image = image_id
                
                # Convert bbox to format expected by MLX SAM3
                # MLX SAM3 expects boxes in format [x0, y0, x1, y1] normalized to [0, 1]
                x1, y1, x2, y2 = bbox[0]
                box_normalized = np.array([[x1 / w, y1 / h, x2 / w, y2 / h]], dtype=np.float32)
                
                # Try different API methods for setting box prompt
                try:
                    # Try set_box_prompt method
                    if hasattr(self.processor, 'set_box_prompt'):
                        state = self.processor.set_box_prompt(box_normalized, self._current_state)
                    elif hasattr(self.processor, 'set_bbox_prompt'):
                        state = self.processor.set_bbox_prompt(box_normalized, self._current_state)
                    else:
                        # Fallback: try to set box directly in state
                        state = self._current_state.copy() if isinstance(self._current_state, dict) else self._current_state
                        if isinstance(state, dict):
                            state['boxes'] = box_normalized
                        else:
                            # If state is not a dict, try to call processor with box
                            state = self.processor(box_normalized, self._current_state)
                except Exception as e:
                    logger.warning(f"Failed to set box prompt with primary method: {e}, trying alternative")
                    # Alternative: try to use the processor directly
                    try:
                        state = self.processor(box_normalized, self._current_state)
                    except:
                        # Last resort: return empty mask
                        logger.error(f"Failed to generate mask from bbox: {e}")
                        return np.zeros((h, w), dtype=bool)
                
                # Get masks from state
                if isinstance(state, dict):
                    masks = state.get("masks", [])
                else:
                    # If state is not a dict, try to get masks attribute
                    masks = getattr(state, "masks", [])
                
                if not masks or len(masks) == 0:
                    return np.zeros((h, w), dtype=bool)
                
                # Convert mask to numpy array
                mask = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
                
                # Ensure mask is boolean and correct shape
                if mask.dtype != bool:
                    mask = mask > 0.5 if mask.max() <= 1.0 else mask > 127
                
                # Ensure correct shape
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

def _load_ultralytics_sam3() -> SAM3Model:
    """Load Ultralytics SAM3 model for Windows/Linux."""
    from ultralytics import SAM
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def find_sam3_checkpoint() -> str:
        import glob
        model_path = os.getenv("SAM3_MODEL_PATH")
        if model_path:
            if os.path.exists(model_path):
                return model_path
            model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
            model_base = os.path.basename(model_path) if os.path.basename(model_path) else "sam3.pt"
        else:
            model_dir = "/models" if os.path.exists("/models") else os.path.join(PROJECT_ROOT, "models")
            model_base = "sam3.pt"
        
        full_path = os.path.join(model_dir, model_base)
        if os.path.exists(full_path):
            return full_path
        
        patterns = [
            os.path.join(model_dir, "sam3-*-of-*.pt"),
        ]
        
        for pattern in patterns:
            shards = sorted(glob.glob(pattern))
            if len(shards) >= 2:
                logger.info(f"Found {len(shards)} sharded checkpoint files, merging...")
                merged_path = os.path.join(model_dir, "sam3.pt")
                if not os.path.exists(merged_path):
                    import torch
                    checkpoint = {}
                    for shard in shards:
                        logger.info(f"Loading shard: {shard}")
                        shard_data = torch.load(shard, map_location="cpu")
                        if isinstance(shard_data, dict):
                            checkpoint.update(shard_data)
                        else:
                            checkpoint.update(shard_data.state_dict() if hasattr(shard_data, 'state_dict') else {})
                    logger.info(f"Saving merged checkpoint to: {merged_path}")
                    torch.save(checkpoint, merged_path)
                return merged_path
        
        return full_path
    
    checkpoint_path = find_sam3_checkpoint()
    logger.info(f"Loading Ultralytics SAM3 model from: {checkpoint_path}")
    return SAM(checkpoint_path)

def generate_mask_from_bbox(
    sam3_model: SAM3Model,
    image_pil: Image.Image,
    bbox: np.ndarray,
    platform_type: str,
) -> np.ndarray:
    """
    Unified interface to generate mask from bounding box.
    
    Args:
        sam3_model: SAM3 model (MLX or Ultralytics)
        image_pil: PIL Image
        bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2]
        platform_type: "mlx" or "ultralytics"
    
    Returns:
        Binary mask as numpy array
    """
    if platform_type == "mlx":
        return sam3_model.generate_mask_from_bbox(image_pil, bbox)
    else:
        # Ultralytics SAM
        h, w = image_pil.size[1], image_pil.size[0]
        imgsz = max(h, w)
        imgsz = ((imgsz + 13) // 14) * 14
        
        results = sam3_model(image_pil, bboxes=bbox, imgsz=imgsz, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                if masks.ndim == 3:
                    mask = np.any(masks, axis=0).astype(bool)
                elif masks.ndim == 2:
                    mask = masks.astype(bool)
                else:
                    mask = masks.squeeze()
                    if mask.ndim == 3:
                        mask = np.any(mask, axis=0).astype(bool)
                    elif mask.ndim == 2:
                        mask = mask.astype(bool)
                    else:
                        return np.zeros((h, w), dtype=bool)
                
                if mask.ndim != 2:
                    return np.zeros((h, w), dtype=bool)
                
                if mask.shape != (h, w):
                    import cv2
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                return mask
        
        return np.zeros((h, w), dtype=bool)

