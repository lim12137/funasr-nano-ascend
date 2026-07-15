# =============================================================================
# 补丁文件：verify_sherpa_onnx.py
# 作用：启动/构建期验证脚本（无需 NPU、无需模型文件即可跑）。
#   - 确认 sherpa_onnx 可导入、版本可用
#   - 确认 onnxruntime 已注册 AscendExecutionProvider（证明 Ascend EP 接上）
#   - 若模型 4 文件齐备，尝试构建识别器（不跑推理）
# 退出码非 0 表示验证失败，可作为 CI 闸门。
# 用法：python verify_sherpa_onnx.py
# =============================================================================
import os
import sys
from pathlib import Path

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")


def main() -> int:
    try:
        import sherpa_onnx
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 无法导入 sherpa_onnx: {e}")
        return 1
    print(f"[OK] sherpa_onnx 版本: {getattr(sherpa_onnx, '__version__', 'unknown')}")

    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 无法导入 onnxruntime: {e}")
        return 1

    providers = ort.get_available_providers()
    print(f"[INFO] onnxruntime 可用 providers: {providers}")
    if "AscendExecutionProvider" in providers:
        print("[OK] AscendExecutionProvider 已注册（昇腾 EP 接上）")
    else:
        print("[WARN] 未检测到 AscendExecutionProvider（将回退 CPU 推理）")

    # 模型 4 文件齐备则尝试构建识别器（不推理）
    files = {
        "encoder_adaptor": Path(MODEL_DIR) / os.environ.get("ENC_ADAPTOR", "encoder_adaptor.onnx"),
        "llm": Path(MODEL_DIR) / os.environ.get("LLM", "llm.onnx"),
        "embedding": Path(MODEL_DIR) / os.environ.get("EMBEDDING", "embedding.onnx"),
        "tokenizer": Path(MODEL_DIR) / os.environ.get("TOKENIZER", "tokenizer.model"),
    }
    if all(p.exists() for p in files.values()):
        try:
            sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                encoder_adaptor=str(files["encoder_adaptor"]),
                llm=str(files["llm"]),
                embedding=str(files["embedding"]),
                tokenizer=str(files["tokenizer"]),
                num_threads=int(os.environ.get("SHERPA_NUM_THREADS", "1")),
                provider=os.environ.get("SHERPA_PROVIDER", "cpu"),
            )
            print("[OK] 识别器构建成功（模型文件有效）")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] 识别器构建失败: {e}")
            return 1
    else:
        missing = [n for n, p in files.items() if not p.exists()]
        print(f"[SKIP] 模型文件缺失，跳过识别器构建: {missing}")

    print("[DONE] 验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
