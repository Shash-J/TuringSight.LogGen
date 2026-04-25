from collections import deque
from datetime import timedelta


class FrameBuffer:
    def __init__(self, max_age_sec: int = 1500):
        self.max_age_sec = max_age_sec
        self.items = deque()

    def add(self, timestamp_dt, frame_path: str):
        self.items.append({
            "timestamp_dt": timestamp_dt,
            "frame_path": frame_path
        })
        self.prune(timestamp_dt)

    def prune(self, now_dt):
        cutoff = now_dt - timedelta(seconds=self.max_age_sec)
        while self.items and self.items[0]["timestamp_dt"] < cutoff:
            self.items.popleft()

    def get_between(self, start_dt, end_dt):
        return [
            item for item in self.items
            if start_dt <= item["timestamp_dt"] <= end_dt
        ]

    def latest(self):
        if not self.items:
            return None
        return self.items[-1]