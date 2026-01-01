# Dissector

FFXIV glamour gear segmentation microservice. Segments head, upper, lower, shoes, and hands from character images.

## Overview

Dissector uses SAM3 and Grounding DINO for high-precision gear segmentation. Provides HTTP API for microservice deployment.

## Development

```bash
# Install dependencies
poetry install

# Run CLI
poetry run python -m src.cli --input-dir data/images --output-dir data/outputs

# Run API service
poetry run python -m src.api
```

## Build

### Linux/Windows (CUDA)

```bash
docker build -t dissector:latest .
```

### Mac (Apple Silicon)

```bash
docker build -f Dockerfile.mac -t dissector:mac .
```

**Note**: For best performance on Mac, run natively to utilize MPS acceleration. Docker on Mac may not support MPS.

## Deployment

### Docker

```bash
# Linux/Windows
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e SAM3_MODEL_PATH=/models/sam3.pt \
  dissector:latest

# Mac
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e SAM3_MODEL_PATH=/models/sam3.pt \
  dissector:mac
```

### Kubernetes

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
```

See `k8s/README.md` for detailed deployment instructions.

### Mac Native (Recommended for Mac)

For optimal performance on Apple Silicon, run natively:

```bash
poetry install
poetry run python -m src.api
```

The service automatically detects and uses MPS acceleration if available.

## API

- `GET /health` - Health check
- `POST /segment` - Segment image (multipart/form-data)
  - Query params: `box_threshold` (default: 0.3), `text_threshold` (default: 0.25)
  - Returns: JSON with base64-encoded images (upper, lower, shoes, head, hands)

## Requirements

- Python 3.12+
- PyTorch 2.7.0
- SAM3 >= 0.1.2
- FastAPI
