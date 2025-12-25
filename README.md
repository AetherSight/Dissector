# FFXIV Gear Segmentation (Head / Upper / Lower / Shoes)

面向后续 web 子模块使用的分割包，提供批处理 CLI 与可复用的 Python 接口。

## 目录结构
```
ffxiv_seg/
  ├─ pipeline.py   # 核心分割流程
  ├─ cli.py        # 命令行入口（推荐使用）
  └─ __init__.py
ffxiv_gear_seg_rebuild.py  # 旧入口（将被替换），暂保留
models/
  ├─ sam2.1_hiera_base_plus.pt
  └─ sam2_configs/           # Hydra 配置目录（包含 sam2.1_hiera_b+.yaml 等）
images/                     # 输入图片目录
outputs/                    # 运行后输出；每次运行会自动清空
```

## 依赖
- Python 3.10+
- `torch`, `transformers`, `opencv-python`, `Pillow`, `hydra-core`, `sam2`（需安装与你的 SAM2 代码/权重匹配的版本）

## 模型文件
将 SAM2 权重与配置放在项目根的 `models/` 下：
- 权重：`models/sam2.1_hiera_base_plus.pt`
- 配置：`models/sam2_configs/` 目录下的 `sam2.1_hiera_b+.yaml`（Hydra 配置名仍为 `sam2.1_hiera_b+`）

## 运行方式（推荐）
使用包入口：
```
python -m ffxiv_seg.cli \
  --box-threshold 0.3 \
  --text-threshold 0.25 \
  --input-dir /path/to/images \
  --output-dir /path/to/outputs
```
说明：
- 不指定 input/output 时，默认 `images/` 与 `outputs/`。
- 每次运行前会自动清空 `outputs/`。
- 阈值可按需要调整（提高阈值=更严格，降低=更宽松）。

## 输出
- 最终结果：`outputs/<图片文件名>/upper.jpg`, `lower.jpg`, `shoes.jpg`, `head.jpg`（head 仅供参考）
- 调试可视化：`outputs/debug/<图片文件名>/step*.jpg`

## 当前分割逻辑（pipeline）
1. 检测鞋（shoes）
2. 检测下衣（lower，扣除鞋）
3. 检测头（head，用于上衣扣头，不保存）
4. 检测上衣（upper，扣除下衣/鞋/头）
5. 检测手（hands），从上衣中扣除手部，保留手臂
6. 小连通域清理、闭运算，输出白底裁剪图

## 旧入口
`ffxiv_gear_seg_rebuild.py` 仍在，但建议改用 `python -m ffxiv_seg.cli`。确认无依赖后可删除该旧文件。

