# FunASR-Nano ONNX Ascend

FunASR-Nano 语音识别模型在昇腾 NPU 上的 ONNX 推理服务，基于 FastAPI，模型通过挂载加载。

## 模型准备

从 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) 下载 funasr-nano 预导出 ONNX 模型：

```bash
mkdir -p ./models
# 下载并解压 sherpa-onnx-funasr-nano-int8 模型到 ./models/
# 需要的文件: encoder.onnx, decoder.onnx, detokenizer.onnx, tokens.txt
```

或使用 [FunASR-nano-onnx](https://github.com/Wasser1462/FunASR-nano-onnx) 自行导出。

## 运行

### 昇腾 NPU

```bash
docker pull ghcr.io/lim12137/funasr-nano-ascend:master

docker run -d \
  --name funasr-nano \
  --network host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v $(pwd)/models:/app/models:ro \
  -e ONNX_PROVIDERS=AscendExecutionProvider,CPUExecutionProvider \
  ghcr.io/lim12137/funasr-nano-ascend:master
```

### CPU 模式

```bash
docker run -d \
  --name funasr-nano \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  ghcr.io/lim12137/funasr-nano-ascend:master
```

## 接口

```bash
# 健康检查
curl http://localhost:8000/health

# 语音识别
curl -X POST http://localhost:8000/recognize \
  -F "file=@test.wav" \
  -F "language=zh"

# WebSocket 流式识别
wscat -c ws://localhost:8000/ws/stream
```

## 目录结构

```
├── .github/workflows/build.yml
├── app.py                        ← FastAPI 推理服务 (HTTP + WebSocket)
├── Dockerfile                    ← 多阶段构建，从 vllm-ascend 提取 CANN
├── requirements.txt
├── .dockerignore
└── models/                       ← 挂载，不写入镜像
    ├── encoder.onnx
    ├── decoder.onnx
    ├── detokenizer.onnx
    └── tokens.txt
```
