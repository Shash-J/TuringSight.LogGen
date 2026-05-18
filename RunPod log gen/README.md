# TuringSight Hybrid CV-VLM Surveillance Pipeline

This project implements a **State-of-the-Art Hybrid Event-Driven Surveillance Architecture**. It completely decouples continuous tracking (Computer Vision) from semantic reasoning (Vision-Language Models), ensuring zero missed events, accurate custom memory, and massively scalable inference via vLLM.

## Architecture Overview

Instead of blindly polling a VLM on every frame (which is slow and wastes compute), the system operates in two layers:

1. **The CV Tracking Layer (YOLOv8 + ByteTrack):** Runs in real-time, assigning IDs to people and objects. It maintains a highly structured JSON "Memory State" of the room (e.g., tracking how long someone has been stationary).
2. **The VLM Reasoning Layer (vLLM + Qwen2-VL):** The VLM is only triggered when the CV layer hits a specific heuristic (e.g., "Person has been stationary for 15s"). When triggered, the pipeline bundles the last 15 seconds of keyframes and the CV Memory State into a single request. 

The VLM processes this via a high-throughput **vLLM API**, generating a fused log (CV State + VLM Insight) perfect for downstream RAG (Query LLM) applications.

---

## Project Structure

```
RunPod log gen/
├── configs/
│   ├── pipeline.yaml          # Camera, sampling, trigger rules, and vLLM API config
│   └── prompts.yaml           # VLM prompt templates
├── data/
│   ├── frames/                # Saved JPEG frames organised by date
│   └── logs/                  # JSONL output (fused semantic + system logs)
├── src/
│   ├── cv/
│   │   ├── tracker.py         # YOLOv8 + ByteTrack object tracking & Custom Memory State
│   │   └── trigger_manager.py # Evaluates CV state to fire EventTasks
│   ├── inference/
│   │   ├── model_service.py   # API_VLMService (communicates with external vLLM)
│   │   └── worker.py          # Builds dynamic prompts combining CV state and config
│   ├── rtsp/
│   │   └── reader.py          # RTSPFrameSampler (OpenCV-based, precise timestamps)
│   ├── scheduler/
│   │   ├── frame_buffer.py    # Circular frame buffer
│   │   └── task_builder.py    # Builds EventTasks with coalesced multi-frame input
│   ├── storage/
│   │   └── jsonl_writer.py    # Append-only JSONL log writer
│   └── utils/
│       └── time_utils.py      # UTC helpers
├── main.py                    # ← Runtime entry point (Event-driven loop)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Start the vLLM Server (Backend)
To guarantee high throughput, the VLM must be served using an inference engine. We recommend using **vLLM** with a 4-bit quantized model (AWQ/GPTQ) to save VRAM and increase speed 3x.

On your GPU Node, run:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct-AWQ \
  --quantization awq \
  --port 8000
```

### 2. Run the TuringSight Pipeline (Frontend)
Ensure `ultralytics` and other dependencies are installed:
```bash
pip install -r requirements.txt
```

Verify your `configs/pipeline.yaml` points to the vLLM server:
```yaml
model:
  mode: "api"
  api_url: "http://localhost:8000/v1/chat/completions"
  model_name: "Qwen/Qwen2-VL-7B-Instruct-AWQ"
```

Start the pipeline:
```bash
python main.py
```

## Configuration

All runtime settings live in `configs/pipeline.yaml`:

| Section    | Key                  | Description                        |
|------------|----------------------|------------------------------------|
| `camera`   | `rtsp_url`           | RTSP stream URL |
| `camera`   | `focus_areas`        | List of dynamic rules injected into the prompt (e.g., "Check for mobile phones") |
| `model`    | `mode`               | Set to `api` for the hybrid architecture |
| `mqtt`     | `enabled`            | If `true`, logs are pushed directly to AWS IoT Core |

## Fused Log Output

The output stored in `data/logs/office_logs.jsonl` (and sent to MQTT) is uniquely structured for Query LLMs. It fuses the absolute precision of Computer Vision with the semantic reasoning of the VLM.

**Example Log:**
```json
{
  "user_id": "user1",
  "camera_id": "cam_01",
  "task_type": "event_driven",
  "timestamp": 1779098418,
  "cv_state": {
    "event_type": "stationary_entity",
    "entity_id": 1,
    "time_stationary_sec": 15.2,
    "cv_state": {
      "class": "person",
      "bbox": [100, 200, 300, 400],
      "first_seen": 1779098400.0
    }
  },
  "event": "The tracked person is leaning back in their chair, actively typing on their smartphone instead of looking at their workstation monitors.",
  "object": "Surveillance Log"
}
```
