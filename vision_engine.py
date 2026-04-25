import cv2
from ultralytics import YOLO

class VisionEngine:
    def __init__(self):
        print("[INIT] Loading YOLOv8m for high-accuracy person detection...")
        self.model = YOLO('yolov8m.pt') 
        self.prev_frame = None
        self.motion_threshold = 8000 # Sensitivity for motion

    def get_analysis(self, frame):
        # 1. Pixel-level Motion Detection
        motion_detected = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is not None:
            delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            if cv2.countNonZero(thresh) > self.motion_threshold:
                motion_detected = True
        self.prev_frame = gray

        # 2. Accurate Person Detection
        results = self.model(frame, verbose=False, classes=0, conf=0.4)[0]
        person_count = len(results.boxes)

        # 3. Format result
        if person_count == 0:
            cv_caption = "No persons detected."
        else:
            cv_caption = f"{person_count} person(s) detected."

        return motion_detected, person_count, cv_caption