#!/usr/bin/env python3
"""
批量提取上衣（upper_3）分割结果的脚本。
遍历数据集目录结构 S:\\FFXIV_train_dataset2\\装备名称\\图片.jpg，
并将输出写入镜像目录（默认加上 _upper3 后缀）中。
多进程模式：每个进程独立加载模型，避免线程锁竞争。
"""
import argparse
import base64
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from dissector.pipeline import load_models, process_image, get_device

# 全局模型句柄（子进程内初始化）
_SAM3_MODEL = None
_DEVICE = None
_SAM3_BACKEND: Optional[str] = None


def iter_images(root: Path) -> Iterable[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def decode_and_save(img_b64: str, dest: Path) -> None:
    data = base64.b64decode(img_b64)
    if not data:
        raise ValueError("解码结果为空")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    if not dest.exists() or dest.stat().st_size == 0:
        raise IOError(f"无法写入文件: {dest}")


def init_worker(backend: Optional[str] = None, device_hint: Optional[str] = None):
    """在子进程内初始化模型，避免跨线程共享锁瓶颈。"""
    global _SAM3_MODEL, _DEVICE, _SAM3_BACKEND
    _SAM3_BACKEND = backend
    if device_hint:
        import torch
        _DEVICE = torch.device(device_hint)
    else:
        _DEVICE = get_device()
    _SAM3_MODEL = load_models(device=_DEVICE, sam3_backend=_SAM3_BACKEND)


def process_single_worker(
    image_path: str,
    input_root: str,
    output_root: str,
    overwrite: bool,
) -> Tuple[str, str]:
    global _SAM3_MODEL, _DEVICE
    if _SAM3_MODEL is None or _DEVICE is None:
        init_worker()

    image_path_obj = Path(image_path)
    rel = image_path_obj.relative_to(input_root)
    out_path = Path(output_root) / rel.parent / f"{image_path_obj.stem}_upper3.jpg"
    if out_path.exists() and not overwrite:
        return "skipped", str(out_path)

    result = process_image(str(image_path_obj), _SAM3_MODEL, _DEVICE)
    upper_b64 = result.get("upper_3")
    if not upper_b64:
        raise ValueError("结果中未包含 upper_3")
    decode_and_save(upper_b64, out_path)
    return "processed", str(out_path)


def main():
    parser = argparse.ArgumentParser(description="批量生成 upper_3 分割结果")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(r"S:\FFXIV_train_dataset2"),
        help="输入数据集根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="输出根目录（默认为 input-root 后缀 _upper3）",
    )
    parser.add_argument(
        "--sam3-backend",
        type=str,
        default=None,
        choices=["cuda", "mlx"],
        help="强制指定后端（默认为自动检测）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="强制指定设备，如 cuda 或 cpu（默认自动）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="进程并行数（每个进程独立加载模型，显存占用成倍增加）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如输出已存在则重新生成",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="每处理多少张输出一次进度日志",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("batch_upper3")

    input_root = args.input_root
    output_root = args.output_root or Path(str(input_root) + "_upper3")

    if not input_root.exists():
        logger.error(f"输入目录不存在: {input_root}")
        raise SystemExit(1)

    logger.info(f"输入目录: {input_root}")
    logger.info(f"输出目录: {output_root}")
    logger.info(f"进程数: {args.workers}")

    images = list(iter_images(input_root))
    total = len(images)
    if total == 0:
        logger.error("未找到可处理的图像文件")
        raise SystemExit(1)
    logger.info(f"待处理图片数量: {total}")

    detected_device = get_device()
    logger.info(f"检测设备: {detected_device}")
    device_hint = args.device or (detected_device.type if hasattr(detected_device, "type") else None)
    backend_hint = args.sam3_backend

    processed = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(backend_hint, device_hint),
    ) as executor:
        future_to_path = {
            executor.submit(
                process_single_worker,
                str(img),
                str(input_root),
                str(output_root),
                args.overwrite,
            ): img
            for img in images
        }

        for idx, future in enumerate(as_completed(future_to_path), 1):
            img_path = future_to_path[future]
            try:
                status, _ = future.result()
                if status == "skipped":
                    skipped += 1
                else:
                    processed += 1
            except Exception as e:
                failed += 1
                logger.warning(f"处理失败: {img_path} | 原因: {e}")

            if idx % args.log_interval == 0:
                logger.info(
                    f"进度 {idx}/{total} | 成功 {processed} | 跳过 {skipped} | 失败 {failed}"
                )

    logger.info(
        f"完成 | 总计 {total} | 成功 {processed} | 跳过 {skipped} | 失败 {failed}"
    )


if __name__ == "__main__":
    main()
