"""
SAM3 多后端统一推理接口
支持 Ultralytics (PyTorch/CUDA/CPU) 和 MLX (Apple Silicon) 后端
"""
import os
import platform
import logging
import threading
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
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
    
    def generate_mask_from_text_prompt(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> Optional[np.ndarray]:
        """
        从文本提示生成二进制 mask（可选实现）
        
        Args:
            image_pil: PIL Image 对象
            text_prompt: 文本提示字符串
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        # 默认实现：返回 None，子类需要实现
        return None
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """返回后端名称"""
        pass


class UltralyticsSAM3(SAM3Base):
    """Ultralytics SAM3 实现，支持 CUDA/CPU"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化 Ultralytics SAM3
        
        Args:
            model_path: 模型路径，如果为 None 则自动查找
            device: 设备类型，'cuda' 或 'cpu'
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
    
    def generate_mask_from_text_prompt(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> Optional[np.ndarray]:
        """
        从文本提示生成 mask（Ultralytics 实现）
        
        注意：Ultralytics SAM3 可能不支持文本提示，这里使用一个fallback方法
        如果需要真正的文本提示支持，可能需要使用 DINO + SAM3 的组合
        
        Args:
            image_pil: PIL Image 对象
            text_prompt: 文本提示字符串
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        # Ultralytics SAM3 可能不支持直接的文本提示
        # 这里返回 None，调用方可以使用其他方法（如 DINO + SAM3）
        logger.warning("Ultralytics SAM3 does not support text prompts directly. Use DINO + SAM3 combination instead.")
        return None
    
    @property
    def backend_name(self) -> str:
        return "ultralytics"


class MLXSAM3(SAM3Base):
    """MLX SAM3 实现，支持 Apple Silicon"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化 MLX SAM3
        
        Args:
            model_path: 模型路径（MLX 版本通常不需要，使用默认模型）
        """
        try:
            # 导入 MLX SAM3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            logger.info("Loading MLX SAM3 model...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
            self._lock = threading.Lock()
            self._current_state = None
            self._current_image_id = None
            self._image_state = None  # 保存原始的 image state，用于每次文本提示
            
        except ImportError as e:
            logger.error(f"Failed to import MLX SAM3: {e}")
            raise RuntimeError("MLX SAM3 is required on macOS. Install with: pip install mlx-sam3") from e
    
    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        从边界框生成 mask（MLX 实现）
        
        Args:
            image_pil: PIL Image 对象
            bbox: numpy array of shape (1, 4) with [x1, y1, x2, y2] 像素坐标
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            try:
                # 检查是否需要设置新图片
                image_id = id(image_pil)
                if self._current_image_id != image_id:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                # 转换 bbox 为 MLX SAM3 格式: [center_x, center_y, width, height] 归一化到 [0, 1]
                # 确保从 numpy array 中提取的值都转换为 Python 原生类型
                x1, y1, x2, y2 = [float(v) for v in bbox[0]]
                w, h = float(w), float(h)
                # 计算并确保所有中间值都是 Python float
                cx = float((x1 + x2) / 2.0 / w)
                cy = float((y1 + y2) / 2.0 / h)
                box_w = float((x2 - x1) / w)
                box_h = float((y2 - y1) / h)
                box_normalized = [cx, cy, box_w, box_h]
                
                # 添加几何提示并运行推理
                state = self.processor.add_geometric_prompt(box_normalized, True, self._current_state)
                self._current_state = state
                
                # 从 state 中提取 mask
                if "masks" in state:
                    masks = state["masks"]
                    if masks is not None and len(masks) > 0:
                        # MLX array 转换为 numpy
                        import mlx.core as mx
                        if isinstance(masks, mx.array):
                            mask_np = np.array(masks[0])
                        else:
                            mask_np = np.array(masks[0]) if isinstance(masks, (list, tuple)) else np.array(masks)
                        
                        # 确保是 2D 布尔数组
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0]
                        elif mask_np.ndim > 2:
                            mask_np = mask_np.squeeze()
                        
                        # 转换为布尔类型
                        if mask_np.dtype != bool:
                            mask_np = mask_np > 0.5
                        
                        # 确保尺寸匹配
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
        批量从边界框生成 mask（MLX 实现）
        
        Args:
            image_pil: PIL Image 对象
            bboxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] 像素坐标
        
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
                # 设置图片（如果需要）
                image_id = id(image_pil)
                if self._current_image_id != image_id:
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                # 重置所有提示
                self.processor.reset_all_prompts(self._current_state)
                
                # 添加所有框作为正样本
                w, h = float(w), float(h)  # 确保 w, h 是 Python float
                for box in bboxes:
                    # 确保从 numpy array 中提取的值都转换为 Python 原生类型
                    x1, y1, x2, y2 = [float(v) for v in box]
                    # 计算并确保所有中间值都是 Python float
                    cx = float((x1 + x2) / 2.0 / w)
                    cy = float((y1 + y2) / 2.0 / h)
                    box_w = float((x2 - x1) / w)
                    box_h = float((y2 - y1) / h)
                    box_normalized = [cx, cy, box_w, box_h]
                    
                    # 添加几何提示
                    self._current_state = self.processor.add_geometric_prompt(
                        box_normalized, True, self._current_state
                    )
                
                # 从最终 state 中提取所有 masks 并合并
                if "masks" in self._current_state:
                    masks = self._current_state["masks"]
                    if masks is not None and len(masks) > 0:
                        import mlx.core as mx
                        # 合并所有 masks
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
        从文本提示生成 mask（MLX 实现）
        
        Args:
            image_pil: PIL Image 对象
            text_prompt: 文本提示字符串
        
        Returns:
            Binary mask as numpy array of shape (H, W), dtype=bool, or None if failed
        """
        h, w = image_pil.size[1], image_pil.size[0]
        
        with self._lock:
            try:
                # 检查是否需要设置新图片
                image_id = id(image_pil)
                if self._current_image_id != image_id or self._image_state is None:
                    # 重新设置图片，确保 _image_state 是干净的
                    # 这对于文本提示很重要，因为每次文本提示都应该基于干净的 image state
                    self._image_state = self.processor.set_image(image_pil)
                    self._current_state = self._image_state
                    self._current_image_id = image_id
                
                # 每次文本提示都基于原始的 image state，避免状态被覆盖
                # 这样多个文本提示可以独立处理，然后在外部合并
                # 确保总是使用 _image_state（而不是可能被污染的 _current_state）
                base_state = self._image_state
                state = self.processor.set_text_prompt(text_prompt, base_state)
                # 不更新 _current_state，保持 image state 不变
                
                # 从 state 中提取 mask
                if "masks" in state:
                    masks = state["masks"]
                    if masks is not None and len(masks) > 0:
                        import mlx.core as mx
                        # 合并所有 masks（如果有多个）
                        mask_total = None
                        for mask in masks:
                            if isinstance(mask, mx.array):
                                mask_np = np.array(mask)
                            else:
                                mask_np = np.array(mask)
                            
                            # 确保是 2D 布尔数组
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]
                            elif mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            
                            # 转换为布尔类型
                            if mask_np.dtype != bool:
                                mask_np = mask_np > 0.5
                            
                            # 确保尺寸匹配
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
            device: 设备类型，仅用于 Ultralytics（"cuda", "cpu"）
        
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
            else:
                device = "cpu"
        
        logger.info(f"Creating SAM3 backend: {backend} (device: {device if device else 'N/A'})")
        
        if backend == "mlx":
            return MLXSAM3(model_path=model_path)
        elif backend == "ultralytics":
            return UltralyticsSAM3(model_path=model_path, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'mlx', 'ultralytics'")

