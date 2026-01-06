"""
HTTP API service for Dissector gear segmentation.
"""
import asyncio
import io
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .pipeline import load_models, process_image, get_device, remove_background

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Dissector", version="0.1.0")

processor = None
dino_model = None
sam3_model = None
device = None

import platform

_max_workers = int(os.getenv('MAX_WORKERS', 0))
if _max_workers <= 0:
    if platform.system() == "Darwin":
        _max_workers = 1
    else:
        cpu_count = multiprocessing.cpu_count()
        if torch.cuda.is_available():
            _max_workers = min(cpu_count, 8)
        else:
            _max_workers = min(cpu_count // 2, 4)

executor = ThreadPoolExecutor(max_workers=_max_workers)

logger.info(f"Thread pool initialized with {_max_workers} workers")


@app.on_event("startup")
async def load_models_on_startup():
    """Load models when the service starts."""
    global processor, dino_model, sam3_model, device
    
    device = get_device()
    logger.info(f"Loading models on device: {device}")
    
    try:
        processor, dino_model, sam3_model = load_models(
            device=device
        )
        logger.info(f"Models loaded successfully (SAM3 backend: {sam3_model.backend_name})")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global executor
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # For MLX backend, only sam3_model is needed (processor and dino_model are None)
    if sam3_model and sam3_model.backend_name == "mlx":
        models_loaded = sam3_model is not None
    else:
        models_loaded = all([processor, dino_model, sam3_model])
    
    return {
        "status": "healthy",
        "device": str(device) if device else "unknown",
        "sam3_backend": sam3_model.backend_name if sam3_model else "unknown",
        "models_loaded": models_loaded
    }


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
):
    """
    Segment gear from uploaded image.
    
    Returns base64-encoded images for: upper, lower, shoes, head, hands
    """
    # For MLX backend, only sam3_model is needed
    if not sam3_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if sam3_model.backend_name != "mlx" and not all([processor, dino_model]):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        # Process image directly with PIL Image object
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            process_image,
            image_pil,
            processor,
            dino_model,
            sam3_model,
            device,
            box_threshold,
            text_threshold,
        )
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/remove-background")
async def remove_background_endpoint(
    file: UploadFile = File(...),
):
    """
    Remove background from image, keeping only the person/character.
    
    Returns base64-encoded PNG image with transparent background.
    """
    # For MLX backend, only sam3_model is needed
    if not sam3_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if sam3_model.backend_name != "mlx" and not all([processor, dino_model]):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        # Process image directly with PIL Image object
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            remove_background,
            image_pil,
            processor,
            dino_model,
            sam3_model,
            device,
        )
        
        return JSONResponse(content={"image": result})
    
    except Exception as e:
        logger.error(f"Error removing background: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

