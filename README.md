# Dissector

FFXIV glamour gear segmentation microservice. Segments head, upper, lower, shoes, and hands from character images.

## Overview

Dissector uses SAM3 for high-precision gear segmentation. Provides HTTP API for microservice deployment.

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

## Installation & Deployment

### Installation

The project uses separate `pyproject.toml` files for different platforms:

**Windows/Linux:**
```bash
# Copy Windows/Linux configuration
cp env/windows/pyproject.toml pyproject.toml

# Install dependencies
pip install -e .
# Or using poetry
poetry install
```

**macOS:**
```bash
# Copy macOS configuration
cp env/mac/pyproject.toml pyproject.toml

# Install dependencies
pip install -e .
# Or using poetry
poetry install
```

**Dependencies:**
- **Windows/Linux**: Facebook SAM3, PyTorch 2.7.0 (CUDA 12.6)
- **macOS**: MLX, MLX SAM3 (from GitHub)

### Windows/Linux CUDA Support (Optional)

For CUDA acceleration on Windows/Linux, install CUDA-enabled PyTorch:

```bash
# Install PyTorch with CUDA 12.6 (recommended)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

# Or use other CUDA versions
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**macOS dependencies:**
- Python 3.13+
- MLX framework
- MLX SAM3

**Windows/Linux dependencies:**
- Python 3.12+
- PyTorch 2.7.0 (CUDA 12.6)
- Facebook SAM3

### Automatic Backend Selection

The application automatically selects the backend based on the system platform:
- **macOS**: Automatically uses MLX backend
- **Windows/Linux**: Automatically uses CUDA backend (Facebook SAM3)

No manual configuration required!

### Manual Backend Selection

If you need to manually specify the backend:

```python
from dissector.backend import SAM3Factory

# Force CUDA backend (Windows/Linux)
sam3_model = SAM3Factory.create(backend="cuda", device="cuda")

# Force MLX backend (macOS only)
sam3_model = SAM3Factory.create(backend="mlx")
```

### Docker Deployment

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
  - Returns: JSON with base64-encoded images (upper, lower, shoes, head, hands)

## Requirements

- Python 3.12+
- PyTorch >= 2.0.0 (CUDA 12.6 recommended for Windows/Linux)
- Facebook SAM3 (Windows/Linux) or MLX SAM3 >= 0.19.0 (macOS)
- FastAPI
