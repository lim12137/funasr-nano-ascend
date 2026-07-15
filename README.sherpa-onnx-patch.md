# sherpa-onnx 路线离线补丁包

把 FunASR-Nano 推理从"裸 onnxruntime"切换到 **sherpa-onnx**（标准 fbank + 正确解码 + 昇腾 NPU 卸载）。

**离线部署设计**：补丁镜像**只含"8.5.1 缺失的部分 + 修改过的文件"（纯增量，delta），不重新打包 8.5.1 原始镜像**。部署时在离线服务器上把补丁镜像**合并到 8.5.1 原始镜像**上，得到可运行的服务镜像。

> 本补丁包不修改原仓库任何文件，全部以独立补丁文件交付。

## 文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `Dockerfile.sherpa-onnx-patch` | 增量补丁镜像 | 多阶段：build 阶段基于 8.5.1 编译 sherpa-onnx(Ascend) + 装缺失 Python 包；最终 `FROM scratch` 只拷贝 delta |
| `Dockerfile.sherpa-onnx-merge` | 离线合并 | 在离线服务器上把补丁镜像合并到 8.5.1 原始镜像，产出可运行镜像 |
| `app.sherpa-onnx.py` | 补丁 app | 替换原 `app.py`，用 `sherpa_onnx.OfflineRecognizer.from_funasr_nano` |
| `verify_sherpa_onnx.py` | 验证脚本 | 校验：导入 + `AscendExecutionProvider` 注册 +（可选）识别器构建 |
| `build-sherpa-onnx-patch.yml` | CI | QEMU arm64 构建增量补丁镜像，导出 tar 包（离线交付，不依赖 ghcr 拉取） |

## 增量补丁镜像包含什么

- **缺失部分**：`onnxruntime-cann`（含 Ascend EP 的 onnxruntime）、`sherpa-onnx`（源码编译）、`fastapi`/`uvicorn` 及其依赖
- **修改部分**：`app.sherpa-onnx.py`（覆盖 `/app/app.py`）、`verify_sherpa_onnx.py`

8.5.1 原始镜像里的 CANN、torch、Python 等**不在补丁镜像中**，合并后由 8.5.1 提供。

## 流程

### 1) CI 构建增量补丁镜像（有网环境）

push 到 master 或 `workflow_dispatch` 触发 `build-sherpa-onnx-patch.yml`：
- 构建 arm64 增量补丁镜像
- 导出 `sherpa-onnx-patch.tar` 作为 artifact 下载

### 2) 离线服务器：导入补丁 + 合并到 8.5.1

```bash
# 传输 sherpa-onnx-patch.tar 到离线服务器后：
docker load -i sherpa-onnx-patch.tar

# 合并到 8.5.1 原始镜像（需本地已有 ghcr.io/lim12137/funasr-cann:8.5.1-validate）
docker build -f Dockerfile.sherpa-onnx-merge \
  --build-arg PATCH_IMAGE=sherpa-onnx-patch:latest \
  -t funasr-nano-ascend:sherpa-onnx-merged .

# 校验（无需模型/NPU）：确认 Ascend EP 已注册
docker run --rm funasr-nano-ascend:sherpa-onnx-merged python verify_sherpa_onnx.py
```

### 3) 运行（昇腾 NPU）

```bash
docker run -d --name funasr-nano \
  --network host \
  --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_DIR=/app/models -e SHERPA_PROVIDER=ascend \
  funasr-nano-ascend:sherpa-onnx-merged
```

## 模型文件（FunASR-Nano 在 sherpa-onnx 下是 LLM 架构，需 4 个文件）

挂载目录 `MODEL_DIR` 需包含：

| 文件 | 环境变量覆盖 | 说明 |
|---|---|---|
| `encoder_adaptor.onnx` | `ENC_ADAPTOR` | 编码器适配 |
| `llm.onnx` | `LLM` | LLM 主干 |
| `embedding.onnx` | `EMBEDDING` | 嵌入 |
| `tokenizer.model` | `TOKENIZER` | tokenizer |

> 注意：这与原 `app.py`/旧 README 写的 `encoder.onnx/decoder.onnx/detokenizer.onnx/tokens.txt` **不是同一套**。
> 请从 sherpa-onnx 的 FunASR-Nano 导出获取这 4 个文件。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_DIR` | `/app/models` | 模型挂载目录 |
| `SHERPA_PROVIDER` | `cpu` | 推理后端；昇腾设为 `ascend`（需在硬件验证 EP 名称） |
| `SHERPA_NUM_THREADS` | `1` | 线程数 |
| `SHERPA_LANGUAGE` | 空 | 语言 |
| `SHERPA_ITN` | `1` | 逆文本正则化开关 |

## 已知风险 / 待验证（首次 CI + 硬件）

1. **`SHERPA_PROVIDER=ascend` 是否为 sherpa-onnx 接受的 provider 字符串**需在真机确认；
   以 `verify_sherpa_onnx.py` 打印的 `AscendExecutionProvider` 是否注册为准（该 EP 来自 `onnxruntime-cann`）。
2. **`onnxruntime-cann` wheel 是否含 C++ 头**：已处理；若 cmake 报找不到 `onnxruntime_cxx_api.h`，
   在 `Dockerfile.sherpa-onnx-patch` build 阶段启用"源码取头"备选段（见该文件注释）。
3. **QEMU 模拟 arm64 下源码编译 sherpa-onnx** 较慢，已给 360 分钟上限，仍可能需看日志调参。
