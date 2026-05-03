# ============ Stage 1: 从 vllm-ascend 镜像提取 CANN 运行时 ============
FROM openeuler/vllm-ascend:0.9.0rc2-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts AS cann-extract

# ============ Stage 2: 构建最终推理镜像 ============
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY --from=cann-extract /usr/local/Ascend/ascend-toolkit /usr/local/Ascend/ascend-toolkit

ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
