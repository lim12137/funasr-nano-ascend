import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import onnxruntime as ort

app = FastAPI(title="FunASR-Nano ONNX Service")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

# FunASR-Nano 需要 3 个 ONNX 文件: encoder, decoder, detokenizer
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.onnx")
DETOK_PATH = os.path.join(MODEL_DIR, "detokenizer.onnx")
VOCAB_PATH = os.path.join(MODEL_DIR, "tokens.txt")

encoder_session = None
decoder_session = None
detok_session = None
token_dict = None


def load_model():
    global encoder_session, decoder_session, detok_session, token_dict

    providers = os.environ.get("ONNX_PROVIDERS", "CPUExecutionProvider").split(",")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if Path(ENCODER_PATH).exists():
        encoder_session = ort.InferenceSession(ENCODER_PATH, opts, providers=providers)
        print(f"[INFO] encoder loaded, providers: {encoder_session.get_providers()}")
    else:
        print(f"[WARN] {ENCODER_PATH} not found, skip")
        return

    if Path(DECODER_PATH).exists():
        decoder_session = ort.InferenceSession(DECODER_PATH, opts, providers=providers)
        print(f"[INFO] decoder loaded")

    if Path(DETOK_PATH).exists():
        detok_session = ort.InferenceSession(DETOK_PATH, opts, providers=providers)
        print(f"[INFO] detokenizer loaded")

    if Path(VOCAB_PATH).exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            token_dict = {}
            for i, line in enumerate(f):
                token = line.strip()
                token_dict[i] = token
        print(f"[INFO] vocab loaded, {len(token_dict)} tokens")


def preprocess(audio_path: str, sr: int = 16000) -> np.ndarray:
    audio, orig_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    # Fbank 特征提取 (80维)
    import librosa
    feat = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=160, n_mels=80, fmax=8000)
    feat = np.log(np.maximum(feat, 1e-10))
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    return feat.T.astype(np.float32)


def greedy_decode(encoder_out: np.ndarray) -> str:
    if decoder_session is None or token_dict is None:
        return ""

    encoder_out = np.expand_dims(encoder_out, axis=0)
    enc_mask = np.ones((1, encoder_out.shape[1]), dtype=np.float32)

    prev_token = np.array([[0]], dtype=np.int64)
    hyp = []
    cache: dict = {}

    for _ in range(512):
        decoder_inputs = {}
        for inp in decoder_session.get_inputs():
            name = inp.name
            if "token" in name or "prev" in name:
                decoder_inputs[name] = prev_token
            elif "mask" in name:
                decoder_inputs[name] = enc_mask
            elif "encoder" in name:
                decoder_inputs[name] = encoder_out
            elif name in cache:
                decoder_inputs[name] = cache[name]
            else:
                decoder_inputs[name] = np.zeros([1, 1] + list(inp.shape[2:]), dtype=np.float32)

        outputs = decoder_session.run(None, decoder_inputs)

        logits = outputs[0]
        new_token = int(np.argmax(logits, axis=-1).flatten()[-1])
        if new_token == 2:
            break

        prev_token = np.array([[new_token]], dtype=np.int64)

        # 更新 cache
        out_names = [o.name for o in decoder_session.get_outputs()]
        for i, oname in enumerate(out_names):
            if "cache" in oname or "state" in oname:
                cache[oname] = outputs[i]

        hyp.append(new_token)

    # detokenize
    if detok_session is not None:
        tokens_arr = np.array([hyp], dtype=np.int64)
        text = detok_session.run(None, {"tokens": tokens_arr})[0]
        if isinstance(text, np.ndarray):
            text = text.flatten().tolist()
            return "".join(str(t) for t in text if t > 0)

    return "".join(token_dict.get(t, "") for t in hyp)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": encoder_session is not None}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...), language: str = Form(default="zh")):
    if encoder_session is None:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    tmp_path = f"/tmp/{file.filename}"
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        feat = preprocess(tmp_path)
        feat_input = np.expand_dims(feat, axis=0).astype(np.float32)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        inputs = {}
        for inp in encoder_session.get_inputs():
            if "len" in inp.name or "length" in inp.name:
                inputs[inp.name] = feat_len
            else:
                inputs[inp.name] = feat_input

        encoder_out = encoder_session.run(None, inputs)[0]
        text = greedy_decode(encoder_out)
        return {"text": text, "language": language}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    if encoder_session is None:
        await websocket.send_json({"error": "model not loaded"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            tmp_path = "/tmp/ws_chunk.wav"
            with open(tmp_path, "wb") as f:
                f.write(data)
            try:
                feat = preprocess(tmp_path)
                feat_input = np.expand_dims(feat, axis=0).astype(np.float32)
                feat_len = np.array([feat.shape[0]], dtype=np.int32)
                inputs = {}
                for inp in encoder_session.get_inputs():
                    if "len" in inp.name or "length" in inp.name:
                        inputs[inp.name] = feat_len
                    else:
                        inputs[inp.name] = feat_input
                encoder_out = encoder_session.run(None, inputs)[0]
                text = greedy_decode(encoder_out)
                await websocket.send_json({"text": text})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
