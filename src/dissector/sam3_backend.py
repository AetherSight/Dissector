"""
SAM3 多后端统一推理接口
支持 Ultralytics (PyTorch/CUDA/MPS) 和 MLX (Apple Silicon) 后端
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
    """
    Native MLX implementation using mlx-community/sam3-image (likely SAM2 architecture).
    Includes manual pipeline for Preprocessing -> Encoder -> Decoder.
    """
    _gpu_lock = threading.Lock() # Global lock for MLX GPU access safety if needed

    def __init__(self, model_path: str = "mlx-community/sam2.1-hiera-large"):
        self.model, self.processor = self._load_model(model_path)
        logger.info("MLX SAM model loaded successfully.")

    def _load_model(self, repo_id):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        return model, processor

    def _preprocess(self, image_pil: Image.Image, target_size=1024):
        """Resize and normalize image for SAM"""
        import mlx.core as mx
        
        img = np.array(image_pil)
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = cv2.resize(img, (new_w, new_h))
        # Normalize to [0, 1] and float32
        img = (img.astype(np.float32) / 255.0)
        
        # Mean/Std normalization usually done inside SAM2 models or skipped for simple float inputs
        # Standard SAM2 expects: (1, 3, 1024, 1024) typically, but MLX vision handles HWC usually.
        # Check specific model implementation. Assuming HWC -> 1HWC conversion happens in model or here.
        
        img_mx = mx.array(img)
        # Add batch dimension if needed (1, H, W, 3)
        img_mx = mx.expand_dims(img_mx, 0) 
        
        return img_mx, scale, (h, w)

    def _run_inference(self, image_pil, bboxes):
        """Core inference logic"""
        import mlx.core as mx
        
        # 1. Preprocess
        img_mx, scale, orig_shape = self._preprocess(image_pil)
        
        # 2. Scale BBoxes
        # bboxes: (N, 4)
        boxes_scaled = bboxes * scale
        boxes_mx = mx.array(boxes_scaled)
        
        # 3. Encode Image (Heavy GPU op)
        # Using specific SAM2 API structure
        try:
            # Common MLX SAM2 API pattern
            source_encoded = self.model.image_encoder(img_mx)
            
            # 4. Prompt Encode & Mask Decode
            # This handles the batch of boxes
            # Note: SAM2 usually expects boxes in shape (B, N, 4) where B=1 image
            boxes_mx = mx.expand_dims(boxes_mx, 0) # (1, N, 4)
            
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None,
                boxes=boxes_mx,
                masks=None
            )
            
            low_res_masks, iou_preds, _ = self.model.mask_decoder(
                image_embeddings=source_encoded,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True # We want the 3 output masks to select best
            )
            
            # low_res_masks: (1, N, 3, 256, 256)
            # iou_preds: (1, N, 3)
            return low_res_masks, iou_preds, orig_shape
            
        except AttributeError as e:
            logger.error(f"Model architecture mismatch: {e}")
            raise

    def generate_mask_from_bbox(self, image_pil: Image.Image, bbox: np.ndarray) -> Optional[np.ndarray]:
        # bbox is (1, 4)
        import mlx.core as mx
        
        with self._gpu_lock:
            low_res_masks, iou_preds, orig_shape = self._run_inference(image_pil, bbox)
            mx.eval(low_res_masks, iou_preds) # Force eval

        # Convert to numpy
        masks = np.array(low_res_masks[0, 0]) # (3, 256, 256)
        scores = np.array(iou_preds[0, 0])    # (3,)
        
        # Select best mask based on IoU score
        best_idx = np.argmax(scores)
        best_mask_logits = masks[best_idx]
        
        # Upsample to original size
        return self._postprocess_mask(best_mask_logits, orig_shape)

    def generate_mask_from_bboxes(self, image_pil: Image.Image, bboxes: np.ndarray) -> Optional[np.ndarray]:
        import mlx.core as mx
        
        if len(bboxes) == 0: return None
        
        with self._gpu_lock:
            low_res_masks, iou_preds, orig_shape = self._run_inference(image_pil, bboxes)
            mx.eval(low_res_masks, iou_preds)

        # low_res_masks: (1, N, 3, 256, 256)
        # iou_preds: (1, N, 3)
        
        all_masks_np = np.array(low_res_masks[0]) # (N, 3, 256, 256)
        all_scores_np = np.array(iou_preds[0])    # (N, 3)
        
        final_masks = []
        for i in range(len(bboxes)):
            scores = all_scores_np[i]
            masks = all_masks_np[i]
            best_idx = np.argmax(scores)
            mask_logits = masks[best_idx]
            final_masks.append(mask_logits > 0) # Binarize logits
            
        if not final_masks: return None
        
        # Stack and Union
        batch_masks = np.stack(final_masks) # (N, 256, 256)
        union_mask = np.any(batch_masks, axis=0) # (256, 256)
        
        # Upsample Union Mask
        # We pass binary mask as float/uint8 to postprocess
        return self._postprocess_mask(union_mask.astype(np.float32), orig_shape, is_logit=False)

    def _postprocess_mask(self, mask_data, orig_shape, is_logit=True):
        """Resize 256x256 SAM output back to original image size"""
        h, w = orig_shape
        
        # SAM outputs 256x256
        if is_logit:
            mask_data = mask_data > 0
            
        mask_uint8 = mask_data.astype(np.uint8)
        
        # Resize nearest neighbor to keep sharp edges
        full_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
        return full_mask.astype(bool)

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

