import os
import cv2

from src.utils.time_utils import date_folder, safe_filename_timestamp


class FrameStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_frame(self, frame, timestamp_dt):
        day_dir = os.path.join(self.base_dir, date_folder(timestamp_dt))
        os.makedirs(day_dir, exist_ok=True)

        filename = f"{safe_filename_timestamp(timestamp_dt)}.jpg"
        path = os.path.join(day_dir, filename)

        ok = cv2.imwrite(path, frame)
        if not ok:
            raise RuntimeError(f"Failed to save frame to {path}")

        return path