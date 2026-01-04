"""
SAM3 多后端统一推理接口
支持 Ultralytics (PyTorch/CUDA/MPS) 和 MLX (Apple Silicon) 后端
"""
import os
import platform
import logging
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
    ) -> np.ndarray:
        """
        从边界框生成二进制 mask
        
        Args:
            image_pil: PIL Image 对象
            bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2] 像素坐标
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool
        """
        pass
    
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
        
        # 计算 imgsz，必须是 14 的倍数
        imgsz = max(h, w)
        imgsz = ((imgsz + 13) // 14) * 14
        
        # 调用 Ultralytics SAM
        results = self.model(image_pil, bboxes=bbox, imgsz=imgsz, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                # 处理不同维度的 mask
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
                
                # 确保尺寸匹配
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                return mask
        
        return np.zeros((h, w), dtype=bool)
    
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
        
        # 检查是否需要设置新图像
        image_id = id(image_pil)
        if self._current_image_id != image_id:
            self._current_state = self.processor.set_image(image_pil)
            self._current_image_id = image_id
        
        # 转换 bbox 为 MLX 格式（归一化到 [0, 1]）
        x1, y1, x2, y2 = bbox[0]
        box_normalized = np.array([[x1 / w, y1 / h, x2 / w, y2 / h]], dtype=np.float32)
        
        # 设置 box prompt
        try:
            if hasattr(self.processor, 'set_box_prompt'):
                state = self.processor.set_box_prompt(box_normalized, self._current_state)
            elif hasattr(self.processor, 'set_bbox_prompt'):
                state = self.processor.set_bbox_prompt(box_normalized, self._current_state)
            else:
                # 回退方案
                if isinstance(self._current_state, dict):
                    state = self._current_state.copy()
                    state['boxes'] = box_normalized
                else:
                    state = self.processor(box_normalized, self._current_state)
        except Exception as e:
            logger.warning(f"Failed to set box prompt: {e}, trying alternative")
            try:
                state = self.processor(box_normalized, self._current_state)
            except Exception as e2:
                logger.error(f"Failed to generate mask from bbox: {e2}")
                return np.zeros((h, w), dtype=bool)
        
        # 获取 masks
        if isinstance(state, dict):
            masks = state.get("masks", [])
        else:
            masks = getattr(state, "masks", [])
        
        if not masks or len(masks) == 0:
            return np.zeros((h, w), dtype=bool)
        
        # 转换 mask 为 numpy array
        mask = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
        
        # 确保 mask 是布尔类型
        if mask.dtype != bool:
            mask = mask > 0.5 if mask.max() <= 1.0 else mask > 127
        
        # 确保尺寸匹配
        if mask.shape != (h, w):
            mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8)
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        
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

