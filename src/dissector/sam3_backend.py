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
    """MLX SAM3 实现，与 UltralyticsSAM3 行为完全一致"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化 MLX SAM3

        Args:
            model_path: 模型路径（可选）
            device: 设备类型（兼容接口，无实际作用）
        """
        from sam3 import build_sam3_image_model

        logger.info("Loading MLX SAM3 model...")
        self.model = build_sam3_image_model()
        # Sam3Processor 可能不是可调用的，让我们直接使用模型
        self._lock = threading.Lock()

    def generate_mask_from_bbox(
        self,
        image_pil: Image.Image,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """从边界框生成 mask，直接使用模型API"""
        h, w = image_pil.size[1], image_pil.size[0]

        # 直接使用模型API，避免processor调用问题
        with self._lock:
            result = self._direct_inference(image_pil, bbox)

        if not isinstance(result, dict) or "masks" not in result:
            return None

        masks = np.asarray(result["masks"])
        if masks.ndim == 2:
            mask = masks
        elif masks.ndim == 3 and masks.shape[0] > 0:
            # 选择最佳 mask（与 Ultralytics 一致：优先使用 scores，否则第一个）
            scores = result.get("scores", None)
            if scores is not None and len(scores) == masks.shape[0]:
                scores = np.asarray(scores)
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
            else:
                mask = masks[0]
        else:
            return None

        # 确保是布尔类型
        mask = mask.astype(bool)

        # 调整到原始尺寸
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        return mask

    def generate_mask_from_bboxes(
        self,
        image_pil: Image.Image,
        bboxes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """批量从边界框生成 mask，直接使用模型API"""
        if bboxes.size == 0:
            h, w = image_pil.size[1], image_pil.size[0]
            return np.zeros((h, w), dtype=bool)

        h, w = image_pil.size[1], image_pil.size[0]

        # 直接使用批量API
        with self._lock:
            result = self._direct_batch_inference(image_pil, bboxes)

        if not isinstance(result, dict) or "masks" not in result:
            return None

        masks = np.asarray(result["masks"])

        # 与 Ultralytics 完全一致：使用 np.any 合并所有 mask
        if masks.ndim == 2:
            mask = masks.astype(bool)
        else:
            # 合并所有 mask
            mask = np.any(masks, axis=0).astype(bool)

        if mask.ndim != 2:
            return None

        # 调整到原始尺寸
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        return mask


    def _direct_inference(self, image_pil, bbox):
        """直接使用 MLX 模型 API 的回退方法"""
        import mlx.core as mx

        h, w = image_pil.size[1], image_pil.size[0]

        # 预处理
        img = np.array(image_pil)
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        img = (img.astype(np.float32) / 255.0)
        img_mx = mx.array(img)
        img_mx = mx.expand_dims(img_mx, 0)

        # 缩放 bbox
        bbox_scaled = bbox * scale
        boxes_mx = mx.array(bbox_scaled)
        boxes_mx = mx.expand_dims(boxes_mx, 0)

        # 推理
        with self._lock:
            source_encoded = self.model.image_encoder(img_mx)
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None, boxes=boxes_mx, masks=None
            )
            low_res_masks, iou_preds, _ = self.model.mask_decoder(
                image_embeddings=source_encoded,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            mx.eval(low_res_masks, iou_preds)

        # 转换为 numpy
        masks = np.array(low_res_masks[0, 0])  # (3, 256, 256)
        scores = np.array(iou_preds[0, 0])     # (3,)

        # 选择最佳 mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx] > 0

        # 调整尺寸
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        return {"masks": mask, "scores": scores}

    def _direct_batch_inference(self, image_pil, bboxes):
        """直接使用 MLX 模型 API 的批量回退方法"""
        import mlx.core as mx

        h, w = image_pil.size[1], image_pil.size[0]

        # 预处理
        img = np.array(image_pil)
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        img = (img.astype(np.float32) / 255.0)
        img_mx = mx.array(img)
        img_mx = mx.expand_dims(img_mx, 0)

        # 缩放所有 bboxes
        bboxes_scaled = bboxes * scale
        boxes_mx = mx.array(bboxes_scaled)

        # 批量推理
        with self._lock:
            source_encoded = self.model.image_encoder(img_mx)
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None, boxes=boxes_mx, masks=None
            )
            low_res_masks, iou_preds, _ = self.model.mask_decoder(
                image_embeddings=source_encoded,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            mx.eval(low_res_masks, iou_preds)

        # 转换为 numpy 并合并
        # low_res_masks: (1, N, 3, 256, 256)
        # iou_preds: (1, N, 3)
        masks = np.array(low_res_masks[0])  # (N, 3, 256, 256)
        scores = np.array(iou_preds[0])     # (N, 3)

        # 对每个 bbox 选择最佳 mask，然后合并
        final_masks = []
        for i in range(len(bboxes)):
            bbox_masks = masks[i]  # (3, 256, 256)
            bbox_scores = scores[i]  # (3,)
            best_idx = np.argmax(bbox_scores)
            mask = bbox_masks[best_idx] > 0  # (256, 256)
            final_masks.append(mask)

        # 合并所有 bbox 的 mask
        if final_masks:
            combined_mask = np.any(final_masks, axis=0)  # (256, 256)
        else:
            combined_mask = np.zeros((256, 256), dtype=bool)

        # 调整尺寸
        combined_mask = cv2.resize(combined_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        return {"masks": combined_mask}

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

