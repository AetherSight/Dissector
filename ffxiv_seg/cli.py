import argparse
import os

from .pipeline import run_batch


def main():
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

    run_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        dino_model_name=args.dino_model_name,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )


if __name__ == "__main__":
    main()

