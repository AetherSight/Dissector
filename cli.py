#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

from dissector.pipeline import (
    load_models,
    process_image,
    get_device,
    remove_background,
)
from dissector.segmentation import debug_get_mask
from dissector.constants import BODY_PARTS_PROMPTS


def main():
    parser = argparse.ArgumentParser(description='Dissector CLI - Test image segmentation')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--box-threshold', type=float, default=0.3, help='Box threshold (deprecated, kept for compatibility)')
    parser.add_argument('--text-threshold', type=float, default=0.25, help='Text threshold (deprecated, kept for compatibility)')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--remove-bg', action='store_true', help='Also test background removal')
    parser.add_argument('--debug-part', type=str, nargs='+', 
                        choices=list(BODY_PARTS_PROMPTS.keys()),
                        help='Debug mode: generate mask images for all prompts of specified part(s). Can specify multiple parts: --debug-part upper lower head')
    parser.add_argument('--debug-dir', type=str, default='./tmp', help='Debug output directory (used with --debug-part)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("Loading models...")
    device = get_device()
    logger.info(f"Device: {device}")
    
    try:
        processor, dino_model, sam3_model = load_models(
            device=device
        )
        logger.info(f"SAM3 backend: {sam3_model.backend_name}")
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)
    
    if args.debug_part:
        logger.info(f"Debug mode: generating masks for part(s): {', '.join(args.debug_part)}")
        try:
            image_pil = Image.open(args.image_path)
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
            
            debug_dir = Path(args.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug output directory: {debug_dir}")
            
            for part_name in args.debug_part:
                logger.info(f"Processing part: {part_name}")
                debug_get_mask(
                    part_name=part_name,
                    image_pil=image_pil,
                    sam3_model=sam3_model,
                    debug_dir=str(debug_dir),
                )
            
            logger.info(f"Debug masks saved to: {debug_dir}")
            return
        
        except Exception as e:
            logger.error(f"Error in debug mode: {e}", exc_info=True)
            sys.exit(1)
    
    logger.info(f"Processing image: {args.image_path}")
    try:
        image_pil = Image.open(args.image_path)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        results = process_image(
            image_pil,
            processor,
            dino_model,
            sam3_model,
            device,
            args.box_threshold,
            args.text_threshold,
        )
        
        if not results:
            logger.error("process_image returned None or empty results")
            sys.exit(1)
        
        logger.info("Saving segmented parts...")
        import base64
        
        parts = ['upper', 'lower', 'shoes', 'head', 'hands', 'upper_1', 'upper_2', 'upper_3', 'upper_4']
        for part in parts:
            if part in results:
                try:
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
                args.image_path,
                processor,
                dino_model,
                sam3_model,
                device,
            )
            
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

