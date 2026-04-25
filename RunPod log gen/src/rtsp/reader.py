import time
from datetime import datetime, timezone

import cv2

from src.utils.time_utils import utc_iso


class RTSPFrameSampler:
    def __init__(self, rtsp_url: str, reconnect_sec: int = 5, frame_interval_sec: int = 4):
        self.rtsp_url = rtsp_url
        self.reconnect_sec = reconnect_sec
        self.frame_interval_sec = frame_interval_sec
        self.cap = None
        self.last_saved_ts = 0.0

    def connect(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.rtsp_url)
        return self.cap.isOpened()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_loop(self, on_frame_saved, system_logger):
        while True:
            if self.cap is None or not self.cap.isOpened():
                system_logger.write({
                    "timestamp_utc": utc_iso(),
                    "level": "INFO",
                    "component": "rtsp_reader",
                    "event": "stream_connect_attempt",
                    "details": {"rtsp_url": self.rtsp_url}
                })

                connected = self.connect()
                if not connected:
                    system_logger.write({
                        "timestamp_utc": utc_iso(),
                        "level": "ERROR",
                        "component": "rtsp_reader",
                        "event": "stream_connect_failed",
                        "details": {"retry_in_sec": self.reconnect_sec}
                    })
                    time.sleep(self.reconnect_sec)
                    continue

                system_logger.write({
                    "timestamp_utc": utc_iso(),
                    "level": "INFO",
                    "component": "rtsp_reader",
                    "event": "stream_connected",
                    "details": {}
                })

            ok, frame = self.cap.read()
            if not ok or frame is None:
                system_logger.write({
                    "timestamp_utc": utc_iso(),
                    "level": "WARNING",
                    "component": "rtsp_reader",
                    "event": "frame_read_failed",
                    "details": {"retry_in_sec": self.reconnect_sec}
                })
                self.close()
                time.sleep(self.reconnect_sec)
                continue

            now_ts = time.time()
            if now_ts - self.last_saved_ts >= self.frame_interval_sec:
                ts_dt = datetime.now(timezone.utc)
                on_frame_saved(frame, ts_dt)
                self.last_saved_ts = now_ts