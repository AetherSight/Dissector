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
        
        # 每次调用都重新设置图像，避免状态冲突
        # 这对于多个 bbox 的情况更安全
        logger.debug(f"[MLX] Setting image, size: {w}x{h}")
        state = self.processor.set_image(image_pil)
        logger.debug(f"[MLX] Image set, state type: {type(state)}, is_dict: {isinstance(state, dict)}")
        
        # 转换 bbox 为 MLX 格式（归一化到 [0, 1]）
        x1, y1, x2, y2 = bbox[0]
        box_normalized = np.array([[x1 / w, y1 / h, x2 / w, y2 / h]], dtype=np.float32)
        logger.debug(f"[MLX] Bbox normalized: {box_normalized}")
        
        # 尝试不同的 API 方法设置 box prompt
        state_after_box = None
        api_method = None
        
        # 方法1: 尝试 add_geometric_prompt (这是 MLX SAM3 的正确方法)
        if hasattr(self.processor, 'add_geometric_prompt'):
            try:
                logger.debug("[MLX] Trying add_geometric_prompt with box")
                # add_geometric_prompt 可能需要特定的格式，尝试不同的调用方式
                # 格式可能是: add_geometric_prompt(prompt_type, prompt_data, state)
                # 或者: add_geometric_prompt(boxes, state)
                try:
                    # 尝试1: 直接传入 box 和 state
                    state_after_box = self.processor.add_geometric_prompt(box_normalized, state)
                    api_method = "add_geometric_prompt_direct"
                    logger.debug(f"[MLX] add_geometric_prompt (direct) succeeded")
                except Exception as e1:
                    logger.debug(f"[MLX] add_geometric_prompt (direct) failed: {e1}, trying with 'box' type")
                    try:
                        # 尝试2: 传入类型和 box
                        state_after_box = self.processor.add_geometric_prompt("box", box_normalized, state)
                        api_method = "add_geometric_prompt_typed"
                        logger.debug(f"[MLX] add_geometric_prompt (typed) succeeded")
                    except Exception as e2:
                        logger.debug(f"[MLX] add_geometric_prompt (typed) failed: {e2}, trying with dict")
                        try:
                            # 尝试3: 传入字典格式
                            prompt_dict = {"boxes": box_normalized}
                            state_after_box = self.processor.add_geometric_prompt(prompt_dict, state)
                            api_method = "add_geometric_prompt_dict"
                            logger.debug(f"[MLX] add_geometric_prompt (dict) succeeded")
                        except Exception as e3:
                            logger.warning(f"[MLX] All add_geometric_prompt attempts failed: {e1}, {e2}, {e3}")
            except Exception as e:
                logger.warning(f"[MLX] add_geometric_prompt failed: {e}")
        
        # 方法2: 尝试 set_box_prompt (fallback)
        if state_after_box is None and hasattr(self.processor, 'set_box_prompt'):
            try:
                logger.debug("[MLX] Trying set_box_prompt")
                state_after_box = self.processor.set_box_prompt(box_normalized, state)
                api_method = "set_box_prompt"
                logger.debug(f"[MLX] set_box_prompt succeeded")
            except Exception as e:
                logger.debug(f"[MLX] set_box_prompt failed: {e}")
        
        # 方法3: 尝试 set_bbox_prompt (fallback)
        if state_after_box is None and hasattr(self.processor, 'set_bbox_prompt'):
            try:
                logger.debug("[MLX] Trying set_bbox_prompt")
                state_after_box = self.processor.set_bbox_prompt(box_normalized, state)
                api_method = "set_bbox_prompt"
                logger.debug(f"[MLX] set_bbox_prompt succeeded")
            except Exception as e:
                logger.debug(f"[MLX] set_bbox_prompt failed: {e}")
        
        # 方法4: 尝试直接设置 boxes（最后 fallback）
        if state_after_box is None:
            try:
                logger.debug("[MLX] Trying direct box assignment (fallback)")
                if isinstance(state, dict):
                    state_after_box = state.copy()
                    state_after_box['boxes'] = box_normalized
                    api_method = "direct_dict"
                else:
                    state_after_box = state
                    api_method = "direct_state"
                logger.debug(f"[MLX] Direct assignment succeeded")
            except Exception as e:
                logger.warning(f"[MLX] Direct assignment failed: {e}")
        
        if state_after_box is None:
            logger.error("[MLX] All methods failed to set box prompt")
            return np.zeros((h, w), dtype=bool)
        
        logger.debug(f"[MLX] Using API method: {api_method}")
        logger.debug(f"[MLX] State after box keys: {list(state_after_box.keys()) if isinstance(state_after_box, dict) else 'N/A'}")
        
        # 尝试生成 masks - 设置 box 后可能需要调用 processor 或 model 来生成
        masks = None
        
        # 首先检查 state 中是否已经有 masks（add_geometric_prompt 可能已经生成了）
        if isinstance(state_after_box, dict):
            masks = state_after_box.get("masks", [])
            logger.debug(f"[MLX] Initial masks check from dict, count: {len(masks) if masks else 0}")
            if masks:
                logger.debug(f"[MLX] Masks already present in state after {api_method}!")
        
        # 如果还没有 masks，尝试调用 model 来生成
        if not masks or len(masks) == 0:
            logger.debug("[MLX] No masks in state, attempting to generate with model...")
            
            # 方法1: 尝试调用 model 的 o2m_mask_predict 方法
            if hasattr(self.model, 'o2m_mask_predict'):
                try:
                    logger.debug("[MLX] Trying model.o2m_mask_predict()")
                    # o2m_mask_predict 可能需要 state 作为参数
                    result = self.model.o2m_mask_predict(state_after_box)
                    if isinstance(result, dict):
                        masks = result.get("masks", [])
                        logger.debug(f"[MLX] o2m_mask_predict() returned masks, count: {len(masks) if masks else 0}")
                    elif result is not None:
                        # 如果返回的不是 dict，尝试提取 masks
                        if hasattr(result, 'masks'):
                            masks = result.masks
                            logger.debug(f"[MLX] o2m_mask_predict() returned object with masks")
                except Exception as e:
                    logger.debug(f"[MLX] model.o2m_mask_predict() failed: {e}")
            
            # 方法2: 尝试调用 model 的 inst_interactive_predictor 方法
            if (not masks or len(masks) == 0) and hasattr(self.model, 'inst_interactive_predictor'):
                try:
                    logger.debug("[MLX] Trying model.inst_interactive_predictor()")
                    result = self.model.inst_interactive_predictor(state_after_box)
                    if isinstance(result, dict):
                        masks = result.get("masks", [])
                        logger.debug(f"[MLX] inst_interactive_predictor() returned masks, count: {len(masks) if masks else 0}")
                except Exception as e:
                    logger.debug(f"[MLX] model.inst_interactive_predictor() failed: {e}")
            
            # 方法3: 尝试直接调用 model（无参数，使用内部 state）
            if (not masks or len(masks) == 0):
                try:
                    logger.debug("[MLX] Trying model() with no args (using internal state)")
                    # 某些 MLX 模型可能需要通过 processor 来调用
                    # 或者 model 需要从 state 中读取信息
                    if isinstance(state_after_box, dict) and 'backbone_out' in state_after_box:
                        # 尝试使用 state 中的信息调用 model
                        result = self.model()
                        if isinstance(result, dict):
                            masks = result.get("masks", [])
                            logger.debug(f"[MLX] model() returned masks, count: {len(masks) if masks else 0}")
                except Exception as e:
                    logger.debug(f"[MLX] model() failed: {e}")
            
            # 方法4: 尝试通过 processor 调用 model
            if (not masks or len(masks) == 0) and hasattr(self.processor, 'model'):
                try:
                    logger.debug("[MLX] Trying processor.model() with state")
                    # processor.model 可能可以直接处理 state
                    if hasattr(self.processor.model, 'o2m_mask_predict'):
                        result = self.processor.model.o2m_mask_predict(state_after_box)
                        if isinstance(result, dict):
                            masks = result.get("masks", [])
                            logger.debug(f"[MLX] processor.model.o2m_mask_predict() returned masks")
                except Exception as e:
                    logger.debug(f"[MLX] processor.model() approach failed: {e}")
        
        # 如果还是没有 masks，尝试从 state_after_box 中获取
        if not masks or len(masks) == 0:
            if isinstance(state_after_box, dict):
                masks = state_after_box.get("masks", [])
            else:
                masks = getattr(state_after_box, "masks", [])
                if not masks:
                    # 尝试其他可能的属性名
                    for attr in ['mask', 'segmentation', 'output']:
                        if hasattr(state_after_box, attr):
                            potential = getattr(state_after_box, attr)
                            logger.debug(f"[MLX] Found attribute '{attr}': {type(potential)}")
                            if potential is not None:
                                masks = [potential] if not isinstance(potential, (list, tuple)) else potential
                                break
        
        if not masks or len(masks) == 0:
            logger.warning(f"[MLX] No masks found after all attempts. State keys: {list(state_after_box.keys()) if isinstance(state_after_box, dict) else 'N/A'}")
            logger.debug(f"[MLX] Processor methods: {[m for m in dir(self.processor) if not m.startswith('_')]}")
            logger.debug(f"[MLX] Model methods: {[m for m in dir(self.model) if not m.startswith('_')]}")
            return np.zeros((h, w), dtype=bool)
        
        # 转换 mask 为 numpy array
        mask = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
        logger.debug(f"[MLX] Mask converted, dtype: {mask.dtype}, shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")
        
        # 确保 mask 是布尔类型
        if mask.dtype != bool:
            threshold = 0.5 if mask.max() <= 1.0 else 127
            mask = mask > threshold
            logger.debug(f"[MLX] Mask thresholded with {threshold}, new dtype: {mask.dtype}")
        
        # 确保尺寸匹配
        if mask.shape != (h, w):
            logger.debug(f"[MLX] Resizing mask from {mask.shape} to ({h}, {w})")
            mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8)
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        
        mask_sum = np.sum(mask)
        logger.debug(f"[MLX] Final mask: shape={mask.shape}, dtype={mask.dtype}, sum={mask_sum}, coverage={mask_sum/(h*w):.2%}")
        
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

