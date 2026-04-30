# ─── Base: NVIDIA CUDA runtime for RunPod GPU pods ───────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ─── System deps: Python 3.11 + OpenCV headless libs ─────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python deps (cached layer) ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Pre-download model weights at build time ────────────────
# Avoids cold-start download on every pod launch (~900 MB)
RUN python -c "\
from transformers import AutoModelForImageTextToText, AutoProcessor; \
AutoModelForImageTextToText.from_pretrained('LiquidAI/LFM2.5-VL-450M'); \
AutoProcessor.from_pretrained('LiquidAI/LFM2.5-VL-450M')"

# ─── Application code ────────────────────────────────────────
COPY configs/  configs/
COPY src/      src/
COPY main.py   .

# ─── Data dirs (created at build time, mounted at runtime) ───
RUN mkdir -p data/frames data/logs

# ─── Runtime ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
# RTSP_URL is passed at runtime via RunPod env vars
ENV RTSP_URL=""

# E2E TIR Requirement: Expose port 8888 for JupyterLab health check
EXPOSE 8888

# Start JupyterLab so E2E detects it as "Running". 
# You can then open the terminal in the browser and run: python main.py
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
