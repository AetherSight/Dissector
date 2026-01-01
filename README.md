# Dissector

FFXIV glamour gear segmentation microservice. Segments head, upper, lower, shoes, and hands from character images.

## Overview

Dissector uses SAM3 and Grounding DINO for high-precision gear segmentation. Provides HTTP API for microservice deployment.

## Development

```bash
# Install dependencies
poetry install

# Run API service
poetry run python -m dissector
```

## Build

```bash
docker build -t dissector:latest .
```

## Deployment

### Docker

```bash
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e SAM3_MODEL_PATH=/models/sam3.pt \
  dissector:latest
```

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
