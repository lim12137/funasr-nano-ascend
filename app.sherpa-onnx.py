# =============================================================================
# 补丁文件：app.sherpa-onnx.py
# 作用：替换原 app.py（裸 onnxruntime 版，存在 fbank/解码/VAD 多处错误）。
#       改用 sherpa_onnx Python API，自动获得：标准 fbank 特征、正确的
#       FunASR-Nano(LLM) 解码、以及(若 provider=ascend)昇腾 NPU 卸载。
# 用法：在 sherpa-onnx 补丁镜像内 `python app.sherpa-onnx.py` 即可，
#       模型通过挂载目录提供，不写入镜像。
# 注意：本文件不修改任何原有文件，仅为"补丁"；需要时把它作为 /app/app.py 覆盖。
# =============================================================================
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI(title="FunASR-Nano (sherpa-onnx) Ascend Service")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

# FunASR-Nano 在 sherpa-onnx 下是 LLM 架构，需要 4 个文件。
# 文件名可在环境变量覆盖，默认值对应 sherpa-onnx 导出的 FunASR-Nano 模型。
ENC_ADAPTOR = os.path.join(MODEL_DIR, os.environ.get("ENC_ADAPTOR", "encoder_adaptor.onnx"))
LLM = os.path.join(MODEL_DIR, os.environ.get("LLM", "llm.onnx"))
EMBEDDING = os.path.join(MODEL_DIR, os.environ.get("EMBEDDING", "embedding.onnx"))
TOKENIZER = os.path.join(MODEL_DIR, os.environ.get("TOKENIZER", "tokenizer.model"))

# provider: "cpu" 默认；昇腾 NPU 卸载设为 "ascend"（需在硬件上验证 EP 名称）
PROVIDER = os.environ.get("SHERPA_PROVIDER", "cpu")
NUM_THREADS = int(os.environ.get("SHERPA_NUM_THREADS", "1"))
LANGUAGE = os.environ.get("SHERPA_LANGUAGE", "")
ITN = os.environ.get("SHERPA_ITN", "1") == "1"

# 启动验证：打印 onnxruntime 可用 provider，确认 AscendExecutionProvider 是否注册
try:
    import onnxruntime as _ort
    print(f"[INFO] onnxruntime available providers: {_ort.get_available_providers()}")
except Exception as e:  # noqa: BLE001
    print(f"[WARN] cannot import onnxruntime: {e}")

recognizer = None


def load_model():
    global recognizer
    needed = {
        "encoder_adaptor": ENC_ADAPTOR,
        "llm": LLM,
        "embedding": EMBEDDING,
        "tokenizer": TOKENIZER,
    }
    missing = [name for name, p in needed.items() if not Path(p).exists()]
    if missing:
        print(f"[WARN] 缺失模型文件，跳过加载: {missing}")
        return

    recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=ENC_ADAPTOR,
        llm=LLM,
        embedding=EMBEDDING,
        tokenizer=TOKENIZER,
        num_threads=NUM_THREADS,
        provider=PROVIDER,
        language=LANGUAGE,
        itn=ITN,
        decoding_method="greedy_search",
    )
    print(f"[INFO] FunASR-Nano 识别器已加载, provider={PROVIDER}")


def read_audio_as_16k(path: str):
    """读取音频为 16k 单声道 float32，必要时重采样。"""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # 转单声道
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception:  # noqa: BLE001
            raise ValueError(f"音频采样率 {sr} != 16000 且无可用的重采样库(librosa)")
    return audio


def transcribe_file(path: str) -> str:
    if recognizer is None:
        raise RuntimeError("model not loaded")
    samples = read_audio_as_16k(path)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": recognizer is not None, "provider": PROVIDER}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...), language: str = Form(default="")):
    if recognizer is None:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = transcribe_file(tmp_path)
        return {"text": text, "language": language or LANGUAGE}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    if recognizer is None:
        await websocket.send_json({"error": "model not loaded"})
        await websocket.close()
        return

    # 说明：FunASR-Nano 为 LLM 架构，天然按整句离线解码。
    # 这里每收到一条音频即作为整段转录（sherpa-onnx 内部已含正确特征/VAD 友好的管线）。
    # 如需按 VAD 实时切句流式输出，可后续叠加 sherpa_onnx.VoiceActivityDetector。
    try:
        while True:
            data = await websocket.receive_bytes()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                text = transcribe_file(tmp_path)
                await websocket.send_json({"text": text})
            except Exception as e:  # noqa: BLE001
                await websocket.send_json({"error": str(e)})
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
