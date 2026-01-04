# MLX SAM3 macOS 问题诊断

## 问题描述
macOS 下的移除背景和部位切割功能没有工作。

## 可能的问题点

### 1. API 调用问题
**位置**: `src/dissector/sam3_backend.py:197-259`

**问题**:
- 代码尝试了 `set_box_prompt` 和 `set_bbox_prompt`，但可能这些方法不存在或 API 不正确
- 根据 Hugging Face 页面示例，MLX SAM3 主要展示了文本提示的使用，box prompt 的 API 可能不同

**需要验证**:
- MLX SAM3 的实际 API 方法名
- box prompt 的正确调用方式
- 是否需要调用额外的方法来生成 masks

### 2. 状态管理问题
**位置**: `src/dissector/sam3_backend.py:205-209`

**问题**:
- 使用 `image_id = id(image_pil)` 来判断是否需要设置新图像
- 但每次调用 `generate_mask_from_bbox` 时，如果图像相同，会复用状态
- 对于同一个图像的多个 bbox，状态可能会互相干扰

**可能的影响**:
- 第一个 bbox 可能工作，但后续的 bbox 可能失败
- 状态没有正确更新

### 3. Mask 获取问题
**位置**: `src/dissector/sam3_backend.py:236-243`

**问题**:
- 代码假设设置 box prompt 后，state 中会包含 masks
- 但可能 masks 需要单独生成，而不是自动包含在 state 中

**需要验证**:
- state 的结构和内容
- masks 是否在 state 中，还是需要调用其他方法获取
- 是否需要调用 `processor.generate()` 或类似方法

### 4. 错误处理可能隐藏了问题
**位置**: `src/dissector/sam3_backend.py:228-234`

**问题**:
- 如果 API 调用失败，会返回空 mask，但可能没有足够的日志输出
- 警告和错误可能被忽略

## 建议的调试步骤

1. **添加详细日志**:
   - 在 `generate_mask_from_bbox` 中添加日志，记录：
     - state 的类型和内容
     - API 调用是否成功
     - masks 是否存在及其类型
     - mask 的形状和数据类型

2. **检查 MLX SAM3 的实际 API**:
   - 查看 MLX SAM3 的 GitHub 仓库源代码
   - 确认正确的 box prompt API
   - 确认如何获取 masks

3. **测试单个 bbox**:
   - 先测试单个 bbox 是否能生成 mask
   - 再测试多个 bbox 的情况

4. **检查状态管理**:
   - 验证状态是否正确保存和更新
   - 考虑每次调用都重新设置图像，而不是复用状态

## 参考信息

根据 Hugging Face 页面 (https://huggingface.co/mlx-community/sam3-image) 的示例：
```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.5)

image = Image.open("your_image.jpg")
state = processor.set_image(image)

# Segment with text prompt
state = processor.set_text_prompt("person", state)

# Access results
masks = state["masks"]       # Binary segmentation masks
boxes = state["boxes"]       # Bounding boxes [x0, y0, x1, y1]
scores = state["scores"]     # Confidence scores
```

**注意**: 示例只展示了文本提示，没有展示 box prompt 的使用方式。

