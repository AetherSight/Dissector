"""
HTTP API service for Dissector gear segmentation.
"""
import os
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

from .pipeline import load_models, process_image, get_device, remove_background

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Dissector", version="0.1.0")

# Global model instances (loaded on startup)
processor = None
dino_model = None
sam3_processor = None
device = None

# Thread pool for CPU-intensive image processing
# Use CPU count, but limit to reasonable number to avoid resource exhaustion
# If GPU is available, can handle more concurrent requests
_max_workers = int(os.getenv('MAX_WORKERS', 0))
if _max_workers <= 0:
    cpu_count = multiprocessing.cpu_count()
    if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        _max_workers = min(cpu_count, 8)
    else:
        _max_workers = min(cpu_count // 2, 4)
executor = ThreadPoolExecutor(max_workers=_max_workers)
logger.info(f"Thread pool initialized with {_max_workers} workers")


@app.on_event("startup")
async def load_models_on_startup():
    """Load models when the service starts."""
    global processor, dino_model, sam3_processor, device
    
    device = get_device()
    logger.info(f"Loading models on device: {device}")
    
    try:
        processor, dino_model, sam3_processor = load_models(
            dino_model_name="IDEA-Research/grounding-dino-base",
            device=device
        )
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device) if device else "unknown",
        "models_loaded": all([processor, dino_model, sam3_processor])
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
    if not all([processor, dino_model, sam3_processor]):
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
        
        # Save to temporary file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image_pil.save(tmp_file.name, "JPEG")
            tmp_path = tmp_file.name
        
        try:
            # Process image in thread pool to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor,
                process_image,
                tmp_path,
                processor,
                dino_model,
                sam3_processor,
                device,
                box_threshold,
                text_threshold,
            )
            
            return JSONResponse(content=results)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
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
    if not all([processor, dino_model, sam3_processor]):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image_pil.save(tmp_file.name, "JPEG")
            tmp_path = tmp_file.name
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                remove_background,
                tmp_path,
                processor,
                dino_model,
                sam3_processor,
                device,
            )
            
            return JSONResponse(content={"image": result})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error removing background: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

