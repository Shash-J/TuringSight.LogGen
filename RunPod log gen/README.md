# TuringSight LogGen Pipeline

Real-time surveillance log generation using LFM2.5-VL-450M inference on RTSP camera feeds.
Designed to run containerized on RunPod cloud GPU pods.

## Project Structure

```
RunPod log gen/
├── configs/
│   ├── pipeline.yaml          # Camera, sampling, output, and model config
│   └── prompts.yaml           # VLM prompt templates (activity, 3m, 20m)
├── data/
│   ├── frames/                # Saved JPEG frames organised by date
│   └── logs/                  # JSONL output (semantic + system logs)
├── scripts/
│   ├── test_rtsp_capture.py   # Quick RTSP connectivity check
│   ├── test_scheduler.py      # Scheduler integration test (no VLM)
│   └── test_vlm_worker.py     # Full pipeline test (capture → infer → log)
├── src/
│   ├── inference/
│   │   ├── model_service.py   # MockVLMService / RealVLMService (LFM2.5-VL-450M)
│   │   ├── prompts.py         # YAML prompt loader
│   │   └── worker.py          # VLMWorker — runs inference, writes logs
│   ├── rtsp/
│   │   └── reader.py          # RTSPFrameSampler (OpenCV-based)
│   ├── scheduler/
│   │   ├── frame_buffer.py    # Circular frame buffer with time-based pruning
│   │   ├── priority_queue.py  # Priority task queue wrapper
│   │   └── task_builder.py    # Activity / summary task builders
│   ├── storage/
│   │   ├── frame_store.py     # Saves frames to disk with date folders
│   │   └── jsonl_writer.py    # Append-only JSONL log writer
│   └── utils/
│       ├── schema.py          # Log schema helper
│       └── time_utils.py      # UTC helpers, safe-filename timestamps
├── main.py                    # ← Runtime entry point
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Quick Start

### Local (development — mock mode)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Edit configs/pipeline.yaml: set mode to "mock" for testing without GPU
python main.py
```

### Docker Build & Run (local GPU)

```bash
cd "RunPod log gen"

docker build -t turingsight-loggen .

docker run --gpus all \
  -e RTSP_URL="rtsp://admin:Tech%40007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0" \
  -v $(pwd)/data:/app/data \
  turingsight-loggen
```

### Deploy on RunPod (one-shot)

1. **Push image to Docker Hub:**
   ```bash
   docker tag turingsight-loggen YOUR_DOCKERHUB_USER/turingsight-loggen:v1
   docker push YOUR_DOCKERHUB_USER/turingsight-loggen:v1
   ```

2. **Create a GPU Pod on RunPod:**
   - Go to [runpod.io](https://runpod.io) → Pods → Deploy
   - Container image: `YOUR_DOCKERHUB_USER/turingsight-loggen:v1`
   - GPU: Any GPU (RTX 3050+ is sufficient for the 450M model)
   - Environment Variables:
     - `RTSP_URL` = your RTSP stream URL
   - Volume: Mount a network volume to `/app/data` for persistent logs/frames

3. **The pipeline starts automatically** — no manual intervention needed.

> **Note:** The RTSP camera must be reachable from the RunPod pod.
> If your camera is on a private network, set up a VPN tunnel or an
> RTSP relay (e.g. MediaMTX) to expose it publicly.

## Configuration

All runtime settings live in `configs/pipeline.yaml`:

| Section    | Key                  | Description                        |
|------------|----------------------|------------------------------------|
| `camera`   | `rtsp_url`           | RTSP stream URL (overridden by RTSP_URL env var) |
| `camera`   | `reconnect_sec`      | Seconds between reconnect attempts |
| `sampling` | `frame_interval_sec` | Capture interval (seconds)         |
| `model`    | `mode`               | `mock` or `real`                   |
| `model`    | `model_name`         | `LiquidAI/LFM2.5-VL-450M`         |
| `model`    | `device`             | `cuda` / `cpu`                     |
| `model`    | `torch_dtype`        | `bfloat16` (recommended)           |

## Model

**LFM2.5-VL-450M** by Liquid AI — a 450M parameter vision-language model
optimized for edge/real-time workloads. Model weights are pre-downloaded
into the Docker image at build time (~900 MB).

## Log Output

- **Semantic logs** — `data/logs/office_logs.jsonl`
  VLM-generated event + object descriptions per frame / interval.
- **System logs** — `data/logs/system_logs.jsonl`
  Structured operational events (connections, task creation, inference timing).
