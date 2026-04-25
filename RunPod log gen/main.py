"""
TuringSight LogGen Pipeline — main entry point.

Orchestrates:
  1. RTSP frame capture → frame store + circular buffer
  2. Scheduler: per-frame activity tasks + 3 min / 20 min summaries
  3. VLM inference worker (mock or real, set in configs/pipeline.yaml)
  4. Semantic + system JSONL logging
"""

import os
import sys
import yaml
from datetime import timedelta

from src.rtsp.reader import RTSPFrameSampler
from src.storage.frame_store import FrameStore
from src.storage.jsonl_writer import JSONLWriter
from src.utils.time_utils import utc_iso
from src.scheduler.frame_buffer import FrameBuffer
from src.scheduler.task_builder import (
    build_activity_task,
    build_summary_task,
    should_trigger,
)
from src.scheduler.priority_queue import TaskPriorityQueue
from src.inference.prompts import load_prompts
from src.inference.model_service import build_model_service
from src.inference.worker import VLMWorker


def load_config(path: str = "configs/pipeline.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # --- unpack config --------------------------------------------------
    rtsp_url           = cfg["camera"]["rtsp_url"]
    frame_interval_sec = cfg["sampling"]["frame_interval_sec"]
    frame_dir          = cfg["sampling"]["frame_dir"]
    reconnect_sec      = cfg["camera"]["reconnect_sec"]
    system_log_path    = cfg["output"]["system_log_path"]
    semantic_log_path  = cfg["output"]["semantic_log_path"]

    # --- ensure output dirs exist ---------------------------------------
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(system_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(semantic_log_path), exist_ok=True)

    # --- build components -----------------------------------------------
    frame_store     = FrameStore(frame_dir)
    system_logger   = JSONLWriter(system_log_path)
    semantic_logger = JSONLWriter(semantic_log_path)

    frame_buffer = FrameBuffer(max_age_sec=1500)
    task_queue   = TaskPriorityQueue()

    prompts       = load_prompts()
    model_service = build_model_service(cfg["model"])

    worker = VLMWorker(
        cfg=cfg,
        prompts=prompts,
        task_queue=task_queue,
        semantic_logger=semantic_logger,
        system_logger=system_logger,
        model_service=model_service,
    )

    sampler = RTSPFrameSampler(
        rtsp_url=rtsp_url,
        reconnect_sec=reconnect_sec,
        frame_interval_sec=frame_interval_sec,
    )

    # --- summary interval state -----------------------------------------
    last_3m_run  = None
    last_20m_run = None

    # --- helpers --------------------------------------------------------
    def log_task_created(task: dict):
        system_logger.write({
            "timestamp_utc": utc_iso(),
            "level": "INFO",
            "component": "scheduler",
            "event": "task_created",
            "details": {
                "task_id":     task["task_id"],
                "task_type":   task["task_type"],
                "priority":    task["priority"],
                "frame_count": len(task["frame_paths"]),
                "queue_size":  task_queue.qsize(),
            },
        })
        print(
            f"[TASK] {task['task_type']} | "
            f"frames={len(task['frame_paths'])} | "
            f"queue={task_queue.qsize()}"
        )

    def handle_saved_frame(frame, ts_dt):
        nonlocal last_3m_run, last_20m_run

        # 1. persist frame to disk
        frame_path = frame_store.save_frame(frame, ts_dt)
        frame_buffer.add(ts_dt, frame_path)

        system_logger.write({
            "timestamp_utc": utc_iso(ts_dt),
            "level": "INFO",
            "component": "frame_store",
            "event": "frame_saved",
            "details": {"frame_path": frame_path},
        })

        # 2. per-frame activity task
        latest_item = frame_buffer.latest()
        if latest_item:
            activity_task = build_activity_task(
                latest_item, prompt_version="activity_v1"
            )
            task_queue.put(activity_task)
            log_task_created(activity_task)

        now_ts = ts_dt.timestamp()

        # 3. 3-minute summary
        if should_trigger(last_3m_run, now_ts, 180):
            items_3m = frame_buffer.get_between(
                ts_dt - timedelta(seconds=180), ts_dt
            )
            task_3m = build_summary_task(
                task_type="interval_summary_3m",
                end_dt=ts_dt,
                frame_items=items_3m,
                frame_count=5,
                prompt_version="summary3m_v1",
                priority=2,
            )
            if task_3m:
                task_queue.put(task_3m)
                log_task_created(task_3m)
            last_3m_run = now_ts

        # 4. 20-minute summary
        if should_trigger(last_20m_run, now_ts, 1200):
            items_20m = frame_buffer.get_between(
                ts_dt - timedelta(seconds=1200), ts_dt
            )
            task_20m = build_summary_task(
                task_type="detailed_summary_20m",
                end_dt=ts_dt,
                frame_items=items_20m,
                frame_count=10,
                prompt_version="summary20m_v1",
                priority=1,
            )
            if task_20m:
                task_queue.put(task_20m)
                log_task_created(task_20m)
            last_20m_run = now_ts

        # 5. drain queue — run inference for every pending task
        while not task_queue.empty():
            worker.run_once()

    # --- run ------------------------------------------------------------
    print("=" * 60)
    print("  TuringSight LogGen Pipeline")
    print(f"  Model  : {cfg['model']['model_name']} ({cfg['model']['mode']})")
    print(f"  Camera : {cfg['camera']['camera_id']}")
    print(f"  RTSP   : {rtsp_url[:40]}...")
    print("=" * 60)

    try:
        sampler.read_loop(
            on_frame_saved=handle_saved_frame,
            system_logger=system_logger,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
