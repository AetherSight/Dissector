"""
SAM3 多后端统一推理接口
支持 Ultralytics (PyTorch/CUDA/MPS) 和 MLX (Apple Silicon) 后端
"""
import os
import platform
import logging
import threading
from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class SAM3Base(ABC):
    """SAM3 抽象基类，定义统一接口"""
    
    @abstractmethod
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        从边界框生成二进制 mask
        
        Args:
            image_pil: PIL Image 对象
            bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2] 像素坐标
        
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
        批量从边界框生成 mask（可选实现，默认回退到循环调用）
        
        Args:
            image_pil: PIL Image 对象
            bboxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] 像素坐标
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        return None
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """返回后端名称"""
        pass


class UltralyticsSAM3(SAM3Base):
    """Ultralytics SAM3 实现，支持 CUDA/MPS/CPU"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化 Ultralytics SAM3
        
        Args:
            model_path: 模型路径，如果为 None 则自动查找
            device: 设备类型，'cuda', 'mps', 或 'cpu'
        """
        from ultralytics import SAM
        
        if model_path is None:
            model_path = self._find_sam3_checkpoint()
        
        logger.info(f"Loading Ultralytics SAM3 from: {model_path} (device: {device})")
        self.model = SAM(model_path)
        self.device = device
        self._model_path = model_path
        self._lock = threading.Lock()
    
    def _find_sam3_checkpoint(self) -> str:
        """查找 SAM3 模型文件"""
        import glob
        
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
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
        
        # 查找分片模型
        patterns = [os.path.join(model_dir, "sam3-*-of-*.pt")]
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
    
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """从边界框生成 mask（Ultralytics 实现）"""
        h, w = image_pil.size[1], image_pil.size[0]
        
        imgsz = max(h, w)
        imgsz = ((imgsz + 13) // 14) * 14
        
        with self._lock:
            results = self.model(image_pil, bboxes=bbox, imgsz=imgsz, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                if masks.ndim == 3:
                    if masks.shape[0] > 0:
                        if masks.shape[0] > 1:
                            if hasattr(result, 'scores') and result.scores is not None:
                                scores = result.scores.cpu().numpy()
                                if len(scores) == masks.shape[0]:
                                    best_idx = np.argmax(scores)
                                    mask = masks[best_idx].astype(bool)
                                else:
                                    mask = masks[0].astype(bool)
                            else:
                                mask = masks[0].astype(bool)
                        else:
                            mask = masks[0].astype(bool)
                    else:
                        return None
                elif masks.ndim == 2:
                    mask = masks.astype(bool)
                else:
                    mask = masks.squeeze()
                    if mask.ndim == 3:
                        if mask.shape[0] > 1:
                            mask = mask[0].astype(bool)
                        else:
                            mask = mask[0].astype(bool) if mask.shape[0] == 1 else np.any(mask, axis=0).astype(bool)
                    elif mask.ndim == 2:
                        mask = mask.astype(bool)
                    else:
                        return None
                
                if mask.ndim != 2:
                    return None
                
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                return mask
        
        return None
    
    def generate_mask_from_bboxes(
        self,
        image_pil: Image.Image,
        bboxes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """批量从边界框生成 mask（Ultralytics 实现）"""
        if bboxes.size == 0:
            h, w = image_pil.size[1], image_pil.size[0]
            return np.zeros((h, w), dtype=bool)
        
        h, w = image_pil.size[1], image_pil.size[0]
        
        imgsz = max(h, w)
        imgsz = ((imgsz + 13) // 14) * 14
        
        with self._lock:
            results = self.model(image_pil, bboxes=bboxes, imgsz=imgsz, verbose=False)
        
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
                        return None
                
                if mask.ndim != 2:
                    return None
                
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                return mask
        
        return None
    
    @property
    def backend_name(self) -> str:
        return "ultralytics"


class MLXSAM3(SAM3Base):
    """MLX SAM3 实现，针对 Apple Silicon 优化"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化 MLX SAM3
        
        Args:
            model_path: 模型路径，MLX 版本通常自动下载，可以为 None
        """
        try:
            # 尝试导入 MLX SAM3
            try:
                from sam3 import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                try:
                    from mlx_sam3 import build_sam3_image_model
                    from mlx_sam3.model.sam3_image_processor import Sam3Processor
                except ImportError:
                    raise ImportError(
                        "MLX SAM3 not found. Install with: "
                        "pip install mlx-sam3 or from GitHub: "
                        "git clone https://github.com/Deekshith-Dade/mlx_sam3.git && cd mlx_sam3 && pip install -e ."
                    )
            
            logger.info("Loading MLX SAM3 model...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
            
            # 状态管理
            self._current_state = None
            self._current_image_id = None
            
            # 线程安全锁（MLX 支持多线程，但需要保护共享资源）
            self._lock = threading.Lock()
            
        except ImportError as e:
            logger.error(f"Failed to import MLX SAM3: {e}")
            raise RuntimeError("MLX SAM3 is required on macOS") from e
    
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """从边界框生成 mask（MLX 实现）"""
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            state = self.processor.set_image(image_pil)
        
        x1, y1, x2, y2 = bbox[0]
        x1_norm, y1_norm = float(x1 / w), float(y1 / h)
        x2_norm, y2_norm = float(x2 / w), float(y2 / h)
        
        center_x = float((x1_norm + x2_norm) / 2.0)
        center_y = float((y1_norm + y2_norm) / 2.0)
        box_width = float(x2_norm - x1_norm)
        box_height = float(y2_norm - y1_norm)
        
        box_list = [center_x, center_y, box_width, box_height]
        
        try:
            with self._lock:
                state_after_box = self.processor.add_geometric_prompt(box_list, True, state)
            api_method = "add_geometric_prompt"
        except Exception as e:
            logger.error(f"[MLX] add_geometric_prompt failed: {e}", exc_info=True)
            logger.warning("[MLX] Falling back to direct box assignment (masks may not be generated)")
            if isinstance(state, dict):
                state_after_box = state.copy()
                state_after_box['boxes'] = np.array([[x1_norm, y1_norm, x2_norm, y2_norm]], dtype=np.float32)
                api_method = "direct_dict"
            else:
                state_after_box = state
                api_method = "direct_state"
        
        
        if state_after_box is None:
            logger.error("[MLX] All methods failed to set box prompt")
            return np.zeros((h, w), dtype=bool)
        
        masks = None
        
        if isinstance(state_after_box, dict):
            masks_raw = state_after_box.get("masks", None)
            
            if masks_raw is not None:
                try:
                    if hasattr(masks_raw, '__array__'):
                        masks = np.array(masks_raw)
                    elif isinstance(masks_raw, (list, tuple)):
                        masks = [np.array(m) if hasattr(m, '__array__') else m for m in masks_raw]
                    else:
                        masks = np.array(masks_raw)
                    
                    mask_count = len(masks)
                    if mask_count > 0:
                        logger.info(f"[MLX] Generated {mask_count} mask(s)")
                except Exception as e:
                    logger.error(f"[MLX] Error converting masks to numpy: {e}", exc_info=True)
                    masks = None
            else:
                masks = None
        
        masks_valid = masks is not None and len(masks) > 0
        
        if not masks_valid:
            if hasattr(self.model, 'o2m_mask_predict'):
                try:
                    result = self.model.o2m_mask_predict(state_after_box)
                    if isinstance(result, dict):
                        masks_raw = result.get("masks", [])
                        masks = np.array(masks_raw) if hasattr(masks_raw, '__array__') else np.array(masks_raw)
                        masks_valid = len(masks) > 0
                    elif result is not None:
                        if hasattr(result, 'masks'):
                            masks = result.masks
                except Exception:
                    pass
            
            if not masks_valid and hasattr(self.model, 'inst_interactive_predictor'):
                try:
                    result = self.model.inst_interactive_predictor(state_after_box)
                    if isinstance(result, dict):
                        masks_raw = result.get("masks", [])
                        masks = np.array(masks_raw) if hasattr(masks_raw, '__array__') else np.array(masks_raw)
                        masks_valid = len(masks) > 0
                except Exception:
                    pass
            
            if not masks_valid:
                try:
                    if isinstance(state_after_box, dict) and 'backbone_out' in state_after_box:
                        result = self.model()
                        if isinstance(result, dict):
                            masks_raw = result.get("masks", [])
                            masks = np.array(masks_raw) if hasattr(masks_raw, '__array__') else np.array(masks_raw)
                            masks_valid = len(masks) > 0
                except Exception:
                    pass
            
            if not masks_valid and hasattr(self.processor, 'model'):
                try:
                    if hasattr(self.processor.model, 'o2m_mask_predict'):
                        result = self.processor.model.o2m_mask_predict(state_after_box)
                        if isinstance(result, dict):
                            masks = result.get("masks", [])
                except Exception:
                    pass
        
        # 如果还是没有 masks，尝试从 state_after_box 中获取
        if masks is None:
            if isinstance(state_after_box, dict):
                masks_raw = state_after_box.get("masks", None)
                if masks_raw is not None:
                    # 转换为 numpy array
                    masks = np.array(masks_raw) if hasattr(masks_raw, '__array__') else np.array(masks_raw)
            else:
                masks_raw = getattr(state_after_box, "masks", None)
                if masks_raw is not None:
                    masks = np.array(masks_raw) if hasattr(masks_raw, '__array__') else np.array(masks_raw)
        
        if masks is None or len(masks) == 0:
            logger.warning(f"[MLX] No masks found after all attempts. State keys: {list(state_after_box.keys()) if isinstance(state_after_box, dict) else 'N/A'}")
            return np.zeros((h, w), dtype=bool)
        
        if isinstance(masks, (list, tuple)):
            mask_array = np.array([np.array(m) if hasattr(m, '__array__') else m for m in masks])
        else:
            mask_array = np.array(masks) if hasattr(masks, '__array__') else np.array(masks)
        
        if mask_array.ndim == 3:
            scores = None
            if isinstance(state_after_box, dict):
                scores_raw = state_after_box.get("scores", None)
                if scores_raw is not None:
                    scores = np.array(scores_raw) if hasattr(scores_raw, '__array__') else np.array(scores_raw)
            
            if scores is not None and len(scores) == mask_array.shape[0]:
                best_idx = np.argmax(scores)
                mask = mask_array[best_idx]
            else:
                mask = np.any(mask_array, axis=0).astype(bool)
        elif mask_array.ndim == 2:
            mask = mask_array.astype(bool)
        else:
            while mask_array.ndim > 2:
                mask_array = mask_array[0]
            mask = mask_array.astype(bool)
        
        if mask.ndim == 1:
            mask = mask.reshape((h, w))
        
        while mask.ndim > 2:
            mask = mask[0]
        
        if mask.dtype != bool:
            threshold = 0.5 if mask.max() <= 1.0 else 127
            mask = mask > threshold
        
        if mask.shape != (h, w):
            mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
            if mask_uint8.ndim != 2:
                mask_uint8 = mask_uint8.squeeze()
            mask_pil = Image.fromarray(mask_uint8, mode='L')
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        
        return mask.astype(bool)
    
    def generate_mask_from_bboxes(
        self,
        image_pil: Image.Image,
        bboxes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """批量从边界框生成 mask（MLX 实现）"""
        if bboxes.size == 0:
            h, w = image_pil.size[1], image_pil.size[0]
            return np.zeros((h, w), dtype=bool)
        
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            state = self.processor.set_image(image_pil)
        
        mask_total = None
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1_norm, y1_norm = float(x1 / w), float(y1 / h)
            x2_norm, y2_norm = float(x2 / w), float(y2 / h)
            
            center_x = float((x1_norm + x2_norm) / 2.0)
            center_y = float((y1_norm + y2_norm) / 2.0)
            box_width = float(x2_norm - x1_norm)
            box_height = float(y2_norm - y1_norm)
            
            box_list = [center_x, center_y, box_width, box_height]
            
            try:
                with self._lock:
                    state_after_box = self.processor.add_geometric_prompt(box_list, True, state)
            except Exception:
                continue
            
            if state_after_box is None:
                continue
            
            masks = None
            if isinstance(state_after_box, dict):
                masks_raw = state_after_box.get("masks", None)
                if masks_raw is not None:
                    try:
                        if hasattr(masks_raw, '__array__'):
                            masks = np.array(masks_raw)
                        elif isinstance(masks_raw, (list, tuple)):
                            masks = np.array([np.array(m) if hasattr(m, '__array__') else m for m in masks_raw])
                        else:
                            masks = np.array(masks_raw)
                    except Exception:
                        continue
            
            if masks is None or len(masks) == 0:
                continue
            
            if isinstance(masks, (list, tuple)):
                mask_array = np.array([np.array(m) if hasattr(m, '__array__') else m for m in masks])
            else:
                mask_array = np.array(masks) if hasattr(masks, '__array__') else np.array(masks)
            
            if mask_array.ndim == 3:
                scores = None
                if isinstance(state_after_box, dict):
                    scores_raw = state_after_box.get("scores", None)
                    if scores_raw is not None:
                        scores = np.array(scores_raw) if hasattr(scores_raw, '__array__') else np.array(scores_raw)
                
                if scores is not None and len(scores) == mask_array.shape[0]:
                    best_idx = np.argmax(scores)
                    mask = mask_array[best_idx]
                else:
                    mask = np.any(mask_array, axis=0).astype(bool)
            elif mask_array.ndim == 2:
                mask = mask_array.astype(bool)
            else:
                while mask_array.ndim > 2:
                    mask_array = mask_array[0]
                mask = mask_array.astype(bool)
            
            if mask.ndim != 2 or mask.shape != (h, w):
                continue
            
            if mask.dtype != bool:
                threshold = 0.5 if mask.max() <= 1.0 else 127
                mask = mask > threshold
            
            if mask_total is None:
                mask_total = mask.copy()
            else:
                mask_total |= mask
        
        if mask_total is None:
            return np.zeros((h, w), dtype=bool)
        
        return mask_total.astype(bool)
    
    @property
    def backend_name(self) -> str:
        return "mlx"


class SAM3Factory:
    """SAM3 工厂类，根据配置自动创建对应的后端实例"""
    
    @staticmethod
    def create(
        backend: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> SAM3Base:
        """
        创建 SAM3 后端实例
        
        Args:
            backend: 后端类型，"mlx" 或 "ultralytics"，如果为 None 则自动检测
            model_path: 模型路径（可选）
            device: 设备类型，仅用于 Ultralytics（"cuda", "mps", "cpu"）
        
        Returns:
            SAM3Base 实例
        """
        # 自动检测后端
        if backend is None:
            if platform.system() == "Darwin":
                backend = "mlx"
            else:
                backend = "ultralytics"
        
        # 自动检测设备（仅 Ultralytics）
        if device is None and backend == "ultralytics":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Creating SAM3 backend: {backend} (device: {device if device else 'N/A'})")
        
        if backend == "mlx":
            return MLXSAM3(model_path=model_path)
        elif backend == "ultralytics":
            return UltralyticsSAM3(model_path=model_path, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'mlx', 'ultralytics'")

