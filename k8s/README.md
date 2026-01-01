# Kubernetes Deployment

## 部署步骤

1. **创建 PVC（PersistentVolumeClaim）用于挂载模型文件**
   ```bash
   kubectl apply -f pvc.yaml
   ```

2. **将模型文件复制到 PVC**
   ```bash
   # 方式1: 通过 Pod 临时挂载
   kubectl run -it --rm --image=busybox --restart=Never --overrides='
   {
     "spec": {
       "containers": [{
         "name": "busybox",
         "image": "busybox",
         "command": ["sh"],
         "stdin": true,
         "tty": true,
         "volumeMounts": [{
           "mountPath": "/models",
           "name": "models"
         }]
       }],
       "volumes": [{
         "name": "models",
         "persistentVolumeClaim": {
           "claimName": "dissector-models"
         }
       }]
     }
   }' -- sh
   
   # 在 Pod 内使用 wget 或其他方式下载模型文件到 /models/sam3.pt
   ```

   ```bash
   # 方式2: 如果使用 NFS 或其他共享存储，直接复制文件到挂载点
   ```

3. **部署服务**
   ```bash
   kubectl apply -f deployment.yaml
   ```

4. **检查部署状态**
   ```bash
   kubectl get pods -l app=dissector
   kubectl logs -f deployment/dissector
   ```

5. **测试服务**
   ```bash
   # 端口转发
   kubectl port-forward service/dissector 8000:8000
   
   # 测试健康检查
   curl http://localhost:8000/health
   
   # 测试分割接口
   curl -X POST "http://localhost:8000/segment?box_threshold=0.3&text_threshold=0.25" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
   ```

## GPU 支持

如果需要使用 GPU，取消注释 `deployment.yaml` 中的 GPU 相关配置，并确保：
- 节点已安装 NVIDIA GPU 驱动和 nvidia-container-runtime
- 集群已配置 GPU 资源

## 资源配置

根据实际需求调整 `deployment.yaml` 中的资源请求和限制：
- `requests`: 最小资源需求
- `limits`: 最大资源限制

## 环境变量

- `SAM3_MODEL_PATH`: SAM3 模型文件路径（默认: `/models/sam3.pt`）
- `PYTHONUNBUFFERED`: Python 输出缓冲设置

