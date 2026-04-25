# ─── Build stage ──────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for OpenCV headless + general build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python deps (cached layer) ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ────────────────────────────────────────
COPY configs/  configs/
COPY src/      src/
COPY main.py   .

# ─── Data dirs (created at build time, mounted at runtime) ───
RUN mkdir -p data/frames data/logs

# ─── Runtime ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
CMD ["python", "main.py"]
