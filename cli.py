#!/usr/bin/env python3
"""
CLI tool for testing Dissector pipeline on macOS.
Usage: python cli.py <image_path> [--box-threshold FLOAT] [--text-threshold FLOAT]
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from transformers/huggingface
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from dissector.pipeline import (
    load_models,
    process_image,
    get_device,
    remove_background,
    save_with_white_bg,
    save_debug_overlay,
)
from dissector.sam3_backend import SAM3Factory


def main():
    parser = argparse.ArgumentParser(description='Dissector CLI - Test image segmentation')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--box-threshold', type=float, default=0.3, help='Box threshold for Grounding DINO')
    parser.add_argument('--text-threshold', type=float, default=0.25, help='Text threshold for Grounding DINO')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--remove-bg', action='store_true', help='Also test background removal')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load models
    logger.info("Loading models...")
    device = get_device()
    logger.info(f"Device: {device}")
    
    try:
        processor, dino_model, sam3_model = load_models(
            dino_model_name="IDEA-Research/grounding-dino-base",
            device=device
        )
        logger.info(f"SAM3 backend: {sam3_model.backend_name}")
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)
    
    # Process image segmentation
    logger.info(f"Processing image: {args.image_path}")
    try:
        results = process_image(
            image_path=args.image_path,
            processor=processor,
            dino_model=dino_model,
            sam3_model=sam3_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        
        if not results:
            logger.error("process_image returned None or empty results")
            sys.exit(1)
        
        # Save segmented parts (decode from base64)
        logger.info("Saving segmented parts...")
        import base64
        
        parts = ['upper', 'lower', 'shoes', 'head', 'hands']
        for part in parts:
            if part in results:
                try:
                    # Decode base64
                    img_data = base64.b64decode(results[part])
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        output_path = output_dir / f"{part}.png"
                        cv2.imwrite(str(output_path), img)
                        logger.info(f"Saved: {output_path}")
                    else:
                        logger.warning(f"Failed to decode {part} image")
                except Exception as e:
                    logger.error(f"Error saving {part}: {e}")
        
        logger.info("Segmentation completed")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        sys.exit(1)
    
    # Test background removal if requested
    if args.remove_bg:
        logger.info("Testing background removal...")
        try:
            bg_result = remove_background(
                image_path=args.image_path,
                processor=processor,
                dino_model=dino_model,
                sam3_model=sam3_model,
                device=device,
            )
            
            # Save background removed image
            import base64
            img_data = base64.b64decode(bg_result)
            output_path = output_dir / "background_removed.png"
            with open(output_path, 'wb') as f:
                f.write(img_data)
            logger.info(f"Saved background removed image: {output_path}")
            
        except Exception as e:
            logger.error(f"Error removing background: {e}", exc_info=True)


if __name__ == '__main__':
    main()

