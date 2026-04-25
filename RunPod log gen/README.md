# TuringSight LogGen Pipeline

Real-time surveillance log generation using VLM inference on RTSP camera feeds.

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
│   │   ├── model_service.py   # MockVLMService / RealVLMService factory
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
│       └── time_utils.py      # UTC helpers, safe-filename timestamps
├── main.py                    # ← Runtime entry point
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Quick Start

### Local (development)

```bash
# Create & activate venv (stays outside the container)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Run the pipeline
python main.py
```

### Docker

```bash
docker build -t turingsight-loggen .
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  turingsight-loggen
```

> **Note:** `venv/` is excluded via `.dockerignore` and is never copied into the image.

## Configuration

All runtime settings live in `configs/pipeline.yaml`:

| Section    | Key                | Description                        |
|------------|--------------------|------------------------------------|
| `camera`   | `rtsp_url`         | RTSP stream URL                    |
| `camera`   | `reconnect_sec`    | Seconds between reconnect attempts |
| `sampling` | `frame_interval_sec` | Capture interval (seconds)       |
| `model`    | `mode`             | `mock` or `real`                   |
| `model`    | `model_name`       | HuggingFace model ID               |
| `model`    | `device`           | `cuda` / `cpu`                     |

## Log Output

- **Semantic logs** — `data/logs/office_logs.jsonl`  
  VLM-generated event + object descriptions per frame / interval.
- **System logs** — `data/logs/system_logs.jsonl`  
  Structured operational events (connections, task creation, inference timing).
