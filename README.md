# Dissector

Dissector - FFXIV 穿搭分割（头/上衣/下衣/鞋/手），批处理 CLI，可嵌入后续 web 子模块。

## 快速使用
```
python -m ffxiv_seg.cli \
  --box-threshold 0.3 \
  --text-threshold 0.25 \
  --input-dir data/images \
  --output-dir data/outputs
```
提示：不传入路径时默认使用 `data/images` 作为输入，`data/outputs` 作为输出（运行前会自动清空输出目录）。阈值可按需要调整（高=更严格，低=更宽松）。

## 输入
- 放置待处理图片到 `data/images/`，支持 jpg/png。

## 输出
- 结果：`data/outputs/<图片名>/upper.jpg`, `lower.jpg`, `shoes.jpg`, `head.jpg`（参考）, `hands.jpg`；白底。
- 调试：`data/outputs/debug/<图片名>/step*.jpg`

## 依赖
- Python 3.10+
- torch, transformers, opencv-python, Pillow, hydra-core, sam3（按项目已固定的版本即可）

