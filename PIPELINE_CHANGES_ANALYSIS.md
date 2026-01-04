# Pipeline.py 参数改动分析

## 关键 Commit 对比

### 1. Commit d936d40 (迁移到 Ultralytics) 之前
**`mask_from_boxes` 函数中的关键逻辑：**
```python
for box in boxes:
    x1, y1, x2, y2 = box
    bbox = np.array([[x1, y1, x2, y2]])
    
    # ✅ 计算 imgsz，确保是 14 的倍数
    imgsz = max(h, w)
    imgsz = ((imgsz + 13) // 14) * 14
    results = sam3_model(image_pil, bboxes=bbox, imgsz=imgsz, verbose=False)
    
    # ✅ 详细的 mask 验证和 resize 逻辑
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
                    continue  # ⚠️ 如果处理失败，跳过这个 box
            
            if mask.ndim != 2:
                continue  # ⚠️ 如果维度不对，跳过
            
            # ✅ 如果尺寸不匹配，resize
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            if mask_total is None:
                mask_total = mask.copy()
            else:
                mask_total |= mask
```

### 2. Commit 5b57396 (重构为统一接口) 之后
**`mask_from_boxes` 函数中的关键逻辑：**
```python
for box in boxes:
    x1, y1, x2, y2 = box
    bbox = np.array([[x1, y1, x2, y2]])
    
    # ❌ 移除了 imgsz 参数的计算
    # ❌ 移除了详细的 mask 验证逻辑
    # ✅ 直接调用统一接口
    mask = sam3_model.generate_mask_from_bbox(image_pil, bbox)

    if mask_total is None:
        mask_total = mask.copy()
    else:
        mask_total |= mask
```

### 3. 当前实现 (sam3_backend.py)

**UltralyticsSAM3.generate_mask_from_bbox()：**
```python
# ✅ 仍然有 imgsz 计算
imgsz = max(h, w)
imgsz = ((imgsz + 13) // 14) * 14
results = self.model(image_pil, bboxes=bbox, imgsz=imgsz, verbose=False)

# ✅ 仍然有 mask 验证和 resize
if mask.shape != (h, w):
    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
```

**MLXSAM3.generate_mask_from_bbox()：**
```python
# ❌ 没有 imgsz 参数（MLX SAM3 可能不需要）
# ✅ 有 mask resize 逻辑
if mask.shape != (h, w):
    mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
    if mask_uint8.ndim != 2:
        mask_uint8 = mask_uint8.squeeze()
    mask_pil = Image.fromarray(mask_uint8, mode='L')
    mask_pil = mask_pil.resize((w, h), Image.NEAREST)
    mask = np.array(mask_pil) > 127
```

## 关键差异点

### 1. **`imgsz` 参数**
- **之前**：在 `mask_from_boxes` 中为每个 box 计算 `imgsz`，确保是 14 的倍数
- **现在**：
  - Ultralytics 后端：在 `UltralyticsSAM3.generate_mask_from_bbox()` 中计算
  - MLX 后端：没有 `imgsz` 参数（MLX SAM3 API 可能不需要）

### 2. **Mask 验证和错误处理**
- **之前**：如果 mask 处理失败（维度不对、形状不对），会 `continue` 跳过这个 box，不影响其他 box
- **现在**：
  - Ultralytics 后端：如果失败，返回 `np.zeros((h, w), dtype=bool)`，但不会跳过
  - MLX 后端：如果失败，返回 `np.zeros((h, w), dtype=bool)`，但不会跳过

### 3. **Mask 合并逻辑**
- **之前**：如果某个 box 的 mask 处理失败，会跳过，只合并成功的 mask
- **现在**：所有 box 都会尝试生成 mask，失败的返回全零 mask，然后合并（全零 mask 不影响结果）

## 可能影响效果的因素

1. **MLX 后端没有 `imgsz` 参数**：如果 MLX SAM3 需要特定的图像尺寸处理，可能会影响 mask 质量
2. **错误处理方式改变**：之前会跳过失败的 box，现在会返回全零 mask，理论上应该一致
3. **Mask 选择逻辑**：MLX 后端现在使用 scores 选择最佳 mask，而不是合并所有 mask，这可能影响结果

## 建议检查点

1. **MLX SAM3 是否需要 `imgsz` 参数**：检查 MLX SAM3 的文档或源码
2. **Mask 选择策略**：MLX 后端使用 scores 选择最佳 mask，可能需要对比合并所有 mask 的效果
3. **Mask resize 方法**：MLX 使用 PIL resize，Ultralytics 使用 cv2.resize，可能略有差异

