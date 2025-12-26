import argparse
import os
import logging
import base64

from .pipeline import run_batch


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="FFXIV gear segmentation (head/upper/lower/shoes)")
    parser.add_argument("--dino-model-name", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--input-dir", type=str, default=None, help="Override input dir (default: <project>/images)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir (default: <project>/outputs)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = args.input_dir or os.path.join(project_root, "data", "images")
    output_dir = args.output_dir or os.path.join(project_root, "data", "outputs")

    # clean outputs then run, saving files from returned base64
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = run_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        dino_model_name=args.dino_model_name,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    for idx, res in enumerate(results):
        base_name = str(idx + 1)
        img_out_dir = os.path.join(output_dir, base_name)
        os.makedirs(img_out_dir, exist_ok=True)
        for key, b64 in res.items():
            if not b64:
                continue
            out_path = os.path.join(img_out_dir, f"{key}.jpg")
            with open(out_path, "wb") as f:
                f.write(__import__("base64").b64decode(b64))
            print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()

