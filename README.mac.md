# Mac 运行指南

## 系统要求

- macOS 12.0+ (Monterey 或更高版本)
- Apple Silicon (M1/M2/M3) 或 Intel 芯片
- Python 3.12+

## 安装方式

### 方式 1: 原生运行（推荐，性能最佳）

原生运行可以直接使用 Apple Silicon 的 MPS（Metal Performance Shaders）加速。

```bash
# 安装依赖
poetry install

# 运行 API 服务
poetry run python -m src.api

# 或运行 CLI
poetry run python -m src.cli
```

### 方式 2: Docker 运行

```bash
# 构建镜像（ARM64）
docker build -f Dockerfile.mac -t dissector:mac .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e SAM3_MODEL_PATH=/models/sam3.pt \
  dissector:mac
```

**注意**: Docker 在 Mac 上可能无法直接使用 MPS 加速，建议使用原生运行方式以获得最佳性能。

## 设备检测

代码会自动检测并使用最佳设备：
1. **CUDA** (如果可用，通常用于 Linux/Windows 的 NVIDIA GPU)
2. **MPS** (Apple Silicon 的 Metal 加速)
3. **CPU** (回退选项)

在 Mac 上，如果 PyTorch 支持 MPS，会自动使用 MPS 加速。

## PyTorch 安装

确保安装支持 MPS 的 PyTorch 版本：

```bash
# Apple Silicon
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或者使用 conda
conda install pytorch torchvision -c pytorch
```

## 性能优化

- 使用原生运行而非 Docker 以获得 MPS 加速
- 确保使用最新版本的 PyTorch（支持 MPS）
- 对于大型模型，考虑增加系统内存

## 故障排除

### MPS 不可用

如果 MPS 不可用，检查：
1. PyTorch 版本是否支持 MPS
2. macOS 版本是否 >= 12.0
3. 是否为 Apple Silicon 芯片

代码会自动回退到 CPU 模式。

### Docker 性能问题

如果在 Docker 中运行性能较差，建议：
1. 使用原生运行方式
2. 或使用 `--platform linux/amd64` 运行 x86 版本（但无法使用 MPS）

