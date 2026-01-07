"""
SAM3 multi-backend unified inference interface.
Supports Facebook SAM3 (PyTorch/CUDA/CPU) and MLX (Apple Silicon) backends.
"""
import os
import platform
import logging
import threading
import torch
import numpy as np
import cv2

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from PIL import Image


logger = logging.getLogger(__name__)


class SAM3Base(ABC):
    """SAM3 abstract base class, defines unified interface."""
    
    @abstractmethod
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Generate binary mask from bounding box.
        
        Args:
            image_pil: PIL Image object
            bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2] pixel coordinates
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        pass
    
    def generate_mask_from_bboxes(
        self,
        image_pil: Image.Image,
        bboxes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Batch generate mask from bounding boxes (optional implementation, defaults to loop calls).
        
        Args:
            image_pil: PIL Image object
            bboxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] pixel coordinates
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        return None
    
    def generate_mask_from_text_prompt(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> Optional[np.ndarray]:
        """
        Generate binary mask from text prompt (optional implementation).
        
        Args:
            image_pil: PIL Image object
            text_prompt: Text prompt string
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        return None
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend name."""
        pass


class CUDASAM3(SAM3Base):
    """Facebook SAM3 implementation, supports CUDA/CPU (PyTorch)."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize Facebook SAM3 (PyTorch/CUDA).
        
        Args:
            model_path: Model path (optional, if None will auto-detect)
            device: Device type, 'cuda' or 'cpu'
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import torch
        except ImportError as e:
            logger.error(f"Failed to import Facebook SAM3: {e}")
            raise RuntimeError("Facebook SAM3 is required. Install with: pip install sam3") from e
        
        logger.info(f"Loading Facebook SAM3 (device: {device})")
        
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(PROJECT_ROOT, "models")
        
        bpe_path = os.path.join(models_dir, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            logger.warning(f"BPE file not found at {bpe_path}, using default path")
            bpe_path = None
        
        if model_path is None:
            checkpoint_path = os.path.join(models_dir, "sam3.pt")
            if not os.path.exists(checkpoint_path):
                logger.info(f"Model checkpoint not found at {checkpoint_path}, Facebook SAM3 will download it")
                checkpoint_path = None
        else:
            checkpoint_path = model_path
        if bpe_path and checkpoint_path:
            logger.info(f"Loading model from: {checkpoint_path}")
            logger.info(f"Using BPE file from: {bpe_path}")
            self.model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
        elif bpe_path:
            logger.info(f"Using BPE file from: {bpe_path}")
            self.model = build_sam3_image_model(bpe_path=bpe_path)
        elif checkpoint_path:
            logger.info(f"Loading model from: {checkpoint_path}")
            self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        else:
            logger.info("Using default paths (Facebook SAM3 will download if needed)")
            self.model = build_sam3_image_model()
        
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.model = self.model.cpu()
            self.device = "cpu"
        
        self.processor = Sam3Processor(self.model)
        
        self._lock = threading.Lock()
        self._current_state = None
        self._current_image_id = None
        self._image_state = None  # Store original image state for each text prompt
    
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Generate mask from bounding box (Facebook SAM3 implementation).
        
        Args:
            image_pil: PIL Image object
            bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2] pixel coordinates
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            try:
                image_id = id(image_pil)
                if self._current_image_id != image_id:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                x1, y1, x2, y2 = [float(v) for v in bbox[0]]
                w, h = float(w), float(h)
                cx = float((x1 + x2) / 2.0 / w)
                cy = float((y1 + y2) / 2.0 / h)
                box_w = float((x2 - x1) / w)
                box_h = float((y2 - y1) / h)
                box_normalized = [cx, cy, box_w, box_h]
                
                state = self.processor.add_geometric_prompt(box_normalized, True, self._current_state)
                self._current_state = state
                
                if "masks" in state:
                    masks = state["masks"]
                    if masks is not None and len(masks) > 0:
                        import torch
                        if isinstance(masks, torch.Tensor):
                            mask_np = masks[0].cpu().numpy()
                        elif isinstance(masks, (list, tuple)):
                            mask_np = masks[0].cpu().numpy() if hasattr(masks[0], 'cpu') else np.array(masks[0])
                        else:
                            mask_np = np.array(masks[0])
                        
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0]
                        elif mask_np.ndim > 2:
                            mask_np = mask_np.squeeze()
                        
                        if mask_np.dtype != bool:
                            mask_np = mask_np > 0.5
                        
                        if mask_np.shape != (h, w):
                            mask_uint8 = (mask_np.astype(np.uint8) * 255) if mask_np.dtype == bool else mask_np.astype(np.uint8)
                            mask_np = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        return mask_np.astype(bool)
                
                return None
                
            except Exception as e:
                logger.error(f"Error generating mask from bbox with Facebook SAM3: {e}", exc_info=True)
                return None
    
    def generate_mask_from_bboxes(
        self,
        image_pil: Image.Image,
        bboxes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Batch generate mask from bounding boxes (Facebook SAM3 implementation)."""
        if bboxes.size == 0:
            h, w = image_pil.size[1], image_pil.size[0]
            return np.zeros((h, w), dtype=bool)
        
        h, w = image_pil.size[1], image_pil.size[0]
        mask_total = None
        
        with self._lock:
            try:
                image_id = id(image_pil)
                if self._current_image_id != image_id:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                if hasattr(self.processor, 'reset_all_prompts'):
                    self.processor.reset_all_prompts(self._current_state)
                
                w, h = float(w), float(h)
                for box in bboxes:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    cx = float((x1 + x2) / 2.0 / w)
                    cy = float((y1 + y2) / 2.0 / h)
                    box_w = float((x2 - x1) / w)
                    box_h = float((y2 - y1) / h)
                    box_normalized = [cx, cy, box_w, box_h]
                    
                    self._current_state = self.processor.add_geometric_prompt(
                        box_normalized, True, self._current_state
                    )
                
                if "masks" in self._current_state:
                    masks = self._current_state["masks"]
                    if masks is not None and len(masks) > 0:
                        for i, mask in enumerate(masks):
                            if isinstance(mask, torch.Tensor):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]
                            elif mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            
                            if mask_np.dtype != bool:
                                mask_np = mask_np > 0.5
                            
                            if mask_np.shape != (h, w):
                                mask_uint8 = (mask_np.astype(np.uint8) * 255) if mask_np.dtype == bool else mask_np.astype(np.uint8)
                                mask_np = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            
                            if mask_total is None:
                                mask_total = mask_np.copy()
                            else:
                                mask_total |= mask_np
                
                if mask_total is None:
                    return np.zeros((h, w), dtype=bool)
                
                return mask_total.astype(bool)
                
            except Exception as e:
                logger.error(f"Error generating mask from bboxes with Facebook SAM3: {e}", exc_info=True)
                return None
    
    def generate_mask_from_text_prompt(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> Optional[np.ndarray]:
        """
        Generate mask from text prompt (Facebook SAM3 implementation).
        
        Args:
            image_pil: PIL Image object
            text_prompt: Text prompt string
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            try:
                image_id = id(image_pil)
                if self._current_image_id != image_id or self._image_state is None:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                base_state = self._image_state
                state = self.processor.set_text_prompt(state=base_state, prompt=text_prompt)
                
                if "masks" in state:
                    masks = state["masks"]
                    if masks is not None and len(masks) > 0:
                        import torch
                        mask_total = None
                        for mask in masks:
                            if isinstance(mask, torch.Tensor):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]
                            elif mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            
                            if mask_np.dtype != bool:
                                mask_np = mask_np > 0.5
                            
                            if mask_np.shape != (h, w):
                                mask_uint8 = (mask_np.astype(np.uint8) * 255) if mask_np.dtype == bool else mask_np.astype(np.uint8)
                                mask_np = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            
                            if mask_total is None:
                                mask_total = mask_np.copy()
                            else:
                                mask_total |= mask_np
                        
                        if mask_total is not None:
                            return mask_total.astype(bool)
                
                return None
                
            except Exception as e:
                logger.error(f"Error generating mask from text prompt with Facebook SAM3: {e}", exc_info=True)
                return None
    
    @property
    def backend_name(self) -> str:
        return "cuda"


class MLXSAM3(SAM3Base):
    """MLX SAM3 implementation, supports Apple Silicon."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MLX SAM3.
        
        Args:
            model_path: Model path (optional, if None will auto-detect)
        """
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            logger.info("Loading MLX SAM3 model...")
            
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(PROJECT_ROOT, "models")
            
            bpe_path = os.path.join(models_dir, "bpe_simple_vocab_16e6.txt.gz")
            if not os.path.exists(bpe_path):
                logger.warning(f"BPE file not found at {bpe_path}, using default path")
                bpe_path = None
            
            if model_path is None:
                checkpoint_path = os.path.join(models_dir, "sam3.pt")
                if not os.path.exists(checkpoint_path):
                    logger.info(f"Model checkpoint not found at {checkpoint_path}, MLX SAM3 will download it")
                    checkpoint_path = None
            else:
                checkpoint_path = model_path
            if bpe_path and checkpoint_path:
                logger.info(f"Loading model from: {checkpoint_path}")
                logger.info(f"Using BPE file from: {bpe_path}")
                self.model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
            elif bpe_path:
                logger.info(f"Using BPE file from: {bpe_path}")
                self.model = build_sam3_image_model(bpe_path=bpe_path)
            elif checkpoint_path:
                logger.info(f"Loading model from: {checkpoint_path}")
                self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            else:
                logger.info("Using default paths (MLX SAM3 will download if needed)")
                self.model = build_sam3_image_model()
            
            self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
            self._lock = threading.Lock()
            self._current_state = None
            self._current_image_id = None
            self._image_state = None  # Store original image state for each text prompt
            
        except ImportError as e:
            logger.error(f"Failed to import MLX SAM3: {e}")
            raise RuntimeError("MLX SAM3 is required on macOS. Install with: pip install mlx-sam3") from e
    
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Generate mask from bounding box (MLX implementation).
        
        Args:
            image_pil: PIL Image object
            bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2] pixel coordinates
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            try:
                image_id = id(image_pil)
                if self._current_image_id != image_id:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                x1, y1, x2, y2 = [float(v) for v in bbox[0]]
                w, h = float(w), float(h)
                cx = float((x1 + x2) / 2.0 / w)
                cy = float((y1 + y2) / 2.0 / h)
                box_w = float((x2 - x1) / w)
                box_h = float((y2 - y1) / h)
                box_normalized = [cx, cy, box_w, box_h]
                
                state = self.processor.add_geometric_prompt(box_normalized, True, self._current_state)
                self._current_state = state
                
                if "masks" in state:
                    masks = state["masks"]
                    if masks is not None and len(masks) > 0:
                        import mlx.core as mx
                        if isinstance(masks, mx.array):
                            mask_np = np.array(masks[0])
                        else:
                            mask_np = np.array(masks[0]) if isinstance(masks, (list, tuple)) else np.array(masks)
                        
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0]
                        elif mask_np.ndim > 2:
                            mask_np = mask_np.squeeze()
                        
                        if mask_np.dtype != bool:
                            mask_np = mask_np > 0.5
                        
                        if mask_np.shape != (h, w):
                            mask_uint8 = (mask_np.astype(np.uint8) * 255) if mask_np.dtype == bool else mask_np.astype(np.uint8)
                            mask_np = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        return mask_np.astype(bool)
                
                return None
                
            except Exception as e:
                logger.error(f"Error generating mask from bbox with MLX SAM3: {e}", exc_info=True)
                return None
    
    def generate_mask_from_bboxes(
        self,
        image_pil: Image.Image,
        bboxes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Batch generate mask from bounding boxes (MLX implementation).
        
        Args:
            image_pil: PIL Image object
            bboxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] pixel coordinates
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        if bboxes.size == 0:
            h, w = image_pil.size[1], image_pil.size[0]
            return np.zeros((h, w), dtype=bool)
        
        h, w = image_pil.size[1], image_pil.size[0]
        mask_total = None
        
        with self._lock:
            try:
                image_id = id(image_pil)
                if self._current_image_id != image_id:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                self.processor.reset_all_prompts(self._current_state)
                
                w, h = float(w), float(h)
                for box in bboxes:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    cx = float((x1 + x2) / 2.0 / w)
                    cy = float((y1 + y2) / 2.0 / h)
                    box_w = float((x2 - x1) / w)
                    box_h = float((y2 - y1) / h)
                    box_normalized = [cx, cy, box_w, box_h]
                    
                    self._current_state = self.processor.add_geometric_prompt(
                        box_normalized, True, self._current_state
                    )
                
                if "masks" in self._current_state:
                    masks = self._current_state["masks"]
                    if masks is not None and len(masks) > 0:
                        import mlx.core as mx
                        for i, mask in enumerate(masks):
                            if isinstance(mask, mx.array):
                                mask_np = np.array(mask)
                            else:
                                mask_np = np.array(mask)
                            
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]
                            elif mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            
                            if mask_np.dtype != bool:
                                mask_np = mask_np > 0.5
                            
                            if mask_np.shape != (h, w):
                                mask_uint8 = (mask_np.astype(np.uint8) * 255) if mask_np.dtype == bool else mask_np.astype(np.uint8)
                                mask_np = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            
                            if mask_total is None:
                                mask_total = mask_np.copy()
                            else:
                                mask_total |= mask_np
                
                if mask_total is None:
                    return np.zeros((h, w), dtype=bool)
                
                return mask_total.astype(bool)
                
            except Exception as e:
                logger.error(f"Error generating mask from bboxes with MLX SAM3: {e}", exc_info=True)
                return None
    
    def generate_mask_from_text_prompt(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> Optional[np.ndarray]:
        """
        Generate mask from text prompt (MLX implementation).
        
        Args:
            image_pil: PIL Image object
            text_prompt: Text prompt string
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            try:
                image_id = id(image_pil)
                if self._current_image_id != image_id or self._image_state is None:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                base_state = self._image_state
                state = self.processor.set_text_prompt(text_prompt, base_state)

                if "masks" in state:
                    masks = state["masks"]
                    if masks is not None and len(masks) > 0:
                        import mlx.core as mx
                        mask_total = None
                        for mask in masks:
                            if isinstance(mask, mx.array):
                                mask_np = np.array(mask)
                            else:
                                mask_np = np.array(mask)
                            
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]
                            elif mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            
                            if mask_np.dtype != bool:
                                mask_np = mask_np > 0.5
                            
                            if mask_np.shape != (h, w):
                                mask_uint8 = (mask_np.astype(np.uint8) * 255) if mask_np.dtype == bool else mask_np.astype(np.uint8)
                                mask_np = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            
                            if mask_total is None:
                                mask_total = mask_np.copy()
                            else:
                                mask_total |= mask_np
                        
                        if mask_total is not None:
                            return mask_total.astype(bool)
                
                return None
                
            except Exception as e:
                logger.error(f"Error generating mask from text prompt with MLX SAM3: {e}", exc_info=True)
                return None
    
    @property
    def backend_name(self) -> str:
        return "mlx"


class SAM3Factory:
    """SAM3 factory class, automatically creates corresponding backend instances based on configuration."""
    
    @staticmethod
    def create(
        backend: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> SAM3Base:
        """
        Create SAM3 backend instance.
        
        Args:
            backend: Backend type, "mlx" or "cuda", if None will auto-detect
            model_path: Model path (optional, Facebook SAM3 will auto-download)
            device: Device type, only for CUDA backend ("cuda", "cpu")
        
        Returns:
            SAM3Base instance
        """
        if backend is None:
            if platform.system() == "Darwin":
                backend = "mlx"
            else:
                backend = "cuda"
        
        if device is None and backend == "cuda":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        logger.info(f"Creating SAM3 backend: {backend} (device: {device if device else 'N/A'})")
        
        if backend == "mlx":
            return MLXSAM3(model_path=model_path)
        elif backend == "cuda":
            return CUDASAM3(model_path=model_path, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'mlx', 'cuda'")

