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
    _gpu_lock = threading.Lock()

    def __init__(self, model_path: str | None = None, **kwargs):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

    @staticmethod
    def _box_to_points(x1, y1, x2, y2, w, h):
        eps = 2
        neg = 4

        points = [
            [x1 + eps, y1 + eps],
            [x2 - eps, y1 + eps],
            [x1 + eps, y2 - eps],
            [x2 - eps, y2 - eps],
            [max(0, x1 - neg), max(0, y1 - neg)],
            [min(w - 1, x2 + neg), min(h - 1, y2 + neg)],
        ]
        labels = [1, 1, 1, 1, 0, 0]
        return points, labels

    def generate_mask_from_bbox(self, image_pil, bbox):
        h, w = image_pil.size[1], image_pil.size[0]
        x1, y1, x2, y2 = bbox[0]

        points, labels = self._box_to_points(x1, y1, x2, y2, w, h)

        with self._gpu_lock:
            if hasattr(self.processor, "predict"):
                result = self.processor.predict(
                    image=image_pil,
                    points=points,
                    labels=labels,
                    multimask_output=True,
                )
            elif hasattr(self.processor, "process"):
                result = self.processor.process(
                    image=image_pil,
                    points=points,
                    labels=labels,
                    multimask_output=True,
                )
            else:
                raise RuntimeError(
                    "Sam3Processor has neither predict nor process method"
                )

        if not isinstance(result, dict):
            return None

        masks = result.get("masks", None)
        if masks is None:
            return None

        masks = np.asarray(masks)

        if masks.ndim == 2:
            mask = masks
        elif masks.ndim == 3 and masks.shape[0] > 0:
            # 行为对齐 Ultralytics：最大面积
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            mask = masks[np.argmax(areas)]
        else:
            return None

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask.astype(bool)

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

