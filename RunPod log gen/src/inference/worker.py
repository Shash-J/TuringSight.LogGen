import json
import time


def extract_json_fields(raw_text: str):
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        return {
            "event": data.get("event", ""),
            "object": data.get("object", "")
        }
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw_text[start:end + 1])
            return {
                "event": data.get("event", ""),
                "object": data.get("object", "")
            }
        except Exception:
            pass

    return {
        "event": raw_text,
        "object": "Unable to parse structured object field from model output."
    }


def build_semantic_log(task, parsed_output, cfg, inference_ms):
    camera_cfg = cfg["camera"]

    return {
        "user_id": camera_cfg["user_id"],
        "device_id": camera_cfg["device_id"],
        "camera_id": camera_cfg["camera_id"],
        "event": parsed_output["event"],
        "object": parsed_output["object"],
        "timestamp": int(time.time()),
        "task_type": task["task_type"],
        "frame_paths": task["frame_paths"],
        "prompt_version": task["prompt_version"],
        "meta": {
            "task_timestamp_utc": task["timestamp_utc"],
            "interval_start_utc": task["interval_start_utc"],
            "interval_end_utc": task["interval_end_utc"],
            "inference_ms": inference_ms
        }
    }


class VLMWorker:
    def __init__(self, cfg, prompts, task_queue, semantic_logger, system_logger, model_service):
        self.cfg = cfg
        self.prompts = prompts
        self.task_queue = task_queue
        self.semantic_logger = semantic_logger
        self.system_logger = system_logger
        self.model_service = model_service

    def run_once(self):
        if self.task_queue.empty():
            return False

        task = self.task_queue.get()
        start = time.time()

        self.system_logger.write({
            "timestamp_utc": task["timestamp_utc"],
            "level": "INFO",
            "component": "vlm_worker",
            "event": "inference_started",
            "details": {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "frame_count": len(task["frame_paths"])
            }
        })

        prompt_text = self.prompts[task["prompt_version"]]
        raw_output = self.model_service.run_inference(task["frame_paths"], prompt_text)
        parsed = extract_json_fields(raw_output)

        inference_ms = int((time.time() - start) * 1000)

        log_record = build_semantic_log(task, parsed, self.cfg, inference_ms)
        self.semantic_logger.write(log_record)

        self.system_logger.write({
            "timestamp_utc": task["timestamp_utc"],
            "level": "INFO",
            "component": "vlm_worker",
            "event": "inference_completed",
            "details": {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "inference_ms": inference_ms
            }
        })

        print(f"[VLM] completed {task['task_type']} in {inference_ms} ms")
        return True