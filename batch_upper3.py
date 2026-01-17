#!/usr/bin/env python3
"""
批量提取上衣（upper_3）分割结果的脚本。
遍历数据集目录结构 S:\\FFXIV_train_dataset2\\装备名称\\图片.jpg，
并将输出写入镜像目录（默认加上 _upper3 后缀）中。
"""
import argparse
import base64
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dissector.pipeline import load_models, process_image, get_device


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


def process_single(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    sam3_model,
    device,
    overwrite: bool,
) -> Tuple[str, Path]:
    rel = image_path.relative_to(input_root)
    out_path = output_root / rel.parent / f"{image_path.stem}_upper3.jpg"
    if out_path.exists() and not overwrite:
        return "skipped", out_path

    result = process_image(str(image_path), sam3_model, device)
    upper_b64 = result.get("upper_3")
    if not upper_b64:
        raise ValueError("结果中未包含 upper_3")
    decode_and_save(upper_b64, out_path)
    return "processed", out_path


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
        "--workers",
        type=int,
        default=max(2, min((os.cpu_count() or 8) // 2, 8)),
        help="线程并行数（主要用于流水线并发）",
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
    logger.info(f"并发线程数: {args.workers}")

    images = list(iter_images(input_root))
    total = len(images)
    if total == 0:
        logger.error("未找到可处理的图像文件")
        raise SystemExit(1)
    logger.info(f"待处理图片数量: {total}")

    device = get_device()
    logger.info(f"使用设备: {device}")
    sam3_model = load_models(device=device)
    logger.info(f"SAM3 后端: {sam3_model.backend_name}")

    processed = 0
    skipped = 0
    failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(
                process_single,
                img,
                input_root,
                output_root,
                sam3_model,
                device,
                args.overwrite,
            ): img
            for img in images
        }

        for idx, future in enumerate(as_completed(future_to_path), 1):
            img_path = future_to_path[future]
            try:
                status, _ = future.result()
                with lock:
                    if status == "skipped":
                        skipped += 1
                    else:
                        processed += 1
            except Exception as e:
                with lock:
                    failed += 1
                logger.warning(f"处理失败: {img_path} | 原因: {e}")

            if idx % args.log_interval == 0:
                with lock:
                    logger.info(
                        f"进度 {idx}/{total} | 成功 {processed} | 跳过 {skipped} | 失败 {failed}"
                    )

    logger.info(
        f"完成 | 总计 {total} | 成功 {processed} | 跳过 {skipped} | 失败 {failed}"
    )


if __name__ == "__main__":
    main()
