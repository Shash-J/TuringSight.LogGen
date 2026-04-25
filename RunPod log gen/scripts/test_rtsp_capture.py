import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler.frame_buffer import FrameBuffer
from src.scheduler.task_builder import TaskBuilder
from src.scheduler.priority_queue import PriorityQueue

def test_scheduler():
    print("--- Testing Scheduler Pipeline ---")
    
    # 1. Initialize components
    buffer = FrameBuffer(max_size=10)
    builder = TaskBuilder()
    queue = PriorityQueue()

    # 2. Simulate incoming frames
    print("\n[INFO] Simulating incoming frames...")
    frames_data = [
        {"camera_id": "cam_01", "timestamp": datetime.utcnow().isoformat(), "frame_path": "/data/f1.jpg"},
        {"camera_id": "cam_02", "timestamp": datetime.utcnow().isoformat(), "frame_path": "/data/f2.jpg"},
    ]
    
    for fd in frames_data:
        buffer.add_frame(fd["camera_id"], fd["timestamp"], fd["frame_path"])
        print(f"Added frame to buffer: {fd['camera_id']} - {fd['frame_path']}")

    # 3. Retrieve frames and build tasks
    print("\n[INFO] Building tasks from buffer...")
    latest_frames = buffer.get_latest_frames(count=2)
    
    for f in latest_frames:
        # Give higher priority to cam_01 (lower number = higher priority)
        priority = 1 if f["camera_id"] == "cam_01" else 5
        task = builder.build_task(f, task_type="yolo_detection", priority=priority)
        queue.enqueue(task)
        print(f"Enqueued task | ID: {task['task_id'][:8]} | Priority: {task['priority']} | Camera: {task['camera_id']}")

    # 4. Process tasks from queue based on priority
    print("\n[INFO] Processing queue...")
    while not queue.is_empty():
        task = queue.dequeue()
        print(f"Processing Task | ID: {task['task_id'][:8]} | Priority: {task['priority']} | Camera: {task['camera_id']}")
        time.sleep(0.5)

    print("\n--- Scheduler Test Complete ---")

if __name__ == "__main__":
    test_scheduler()