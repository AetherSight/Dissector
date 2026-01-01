# Dissector

FFXIV 装备分割微服务，支持分割头部、上衣、下衣、鞋子和手部。

## 功能特性

- 使用 SAM3 和 Grounding DINO 进行高精度装备分割
- 支持批量处理图片
- 提供 HTTP API 接口，可作为微服务部署
- 支持 Kubernetes 部署
- 支持 GPU 加速

## 快速使用

### CLI 模式

```bash
python -m src.cli \
  --box-threshold 0.3 \
  --text-threshold 0.25 \
  --input-dir data/images \
  --output-dir data/outputs
```

提示：不传入路径时默认使用 `data/images` 作为输入，`data/outputs` 作为输出（运行前会自动清空输出目录）。阈值可按需要调整（高=更严格，低=更宽松）。

### API 模式

```bash
# 启动 API 服务
python -m src.api

# 健康检查
curl http://localhost:8000/health

# 分割图片
curl -X POST "http://localhost:8000/segment?box_threshold=0.3&text_threshold=0.25" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

## Docker 部署

### 构建镜像

```bash
docker build -t dissector:latest .
```

### 运行容器

```bash
# 挂载模型文件
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e SAM3_MODEL_PATH=/models/sam3.pt \
  dissector:latest
```

## Kubernetes 部署

详细部署说明请参考 [k8s/README.md](k8s/README.md)

### 快速部署

```bash
# 1. 创建 PVC（用于挂载模型）
kubectl apply -f k8s/pvc.yaml

# 2. 将模型文件复制到 PVC（参考 k8s/README.md）

# 3. 部署服务
kubectl apply -f k8s/deployment.yaml

# 4. 检查状态
kubectl get pods -l app=dissector
kubectl logs -f deployment/dissector
```

## 输入输出

### 输入
- 支持格式：JPG, PNG
- CLI 模式：放置图片到 `data/images/`
- API 模式：通过 HTTP POST 上传图片

### 输出
- `upper.jpg` - 上衣
- `lower.jpg` - 下衣
- `shoes.jpg` - 鞋子
- `head.jpg` - 头部（参考）
- `hands.jpg` - 手部
- 所有输出均为白底图片

## 依赖

- Python 3.12+
- PyTorch 2.7.0 (CUDA 12.6)
- SAM3 >= 0.1.2
- FastAPI (API 模式)
- 其他依赖见 `pyproject.toml`

## 模型文件

模型文件需要单独准备：
- SAM3 模型：`models/sam3.pt` 或通过挂载卷提供
- BPE 词汇表：`assets/bpe_simple_vocab_16e6.txt.gz`（已包含）

在 Kubernetes 部署中，模型文件通过 PVC 挂载到 `/models` 目录。

## 开发

```bash
# 安装依赖
poetry install

# 运行测试
poetry run python -m src.cli
```

## License

See LICENSE file for details.
