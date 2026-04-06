import cv2
from ultralytics import YOLO

class VisionEngine:
    def __init__(self):
        # Using YOLOv8 nano (very fast on CPU)
        self.model = YOLO('yolov8n.pt') 
        self.prev_frame = None
        self.motion_threshold = 5000 # Pixel change count

    def get_deterministic_analysis(self, frame):
        """
        Runs continuously. Returns (is_motion, cv_caption)
        """
        # 1. Motion Detection (Pixel level)
        motion_detected = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is not None:
            delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            if cv2.countNonZero(thresh) > self.motion_threshold:
                motion_detected = True
        self.prev_frame = gray

        # 2. Deterministic Captioning (Object Detection)
        results = self.model(frame, verbose=False)[0]
        detections = results.boxes.cls.tolist()
        names = results.names
        
        # Create a summary: "2 persons, 1 dog"
        counts = {}
        for cls_id in detections:
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
        
        if not counts:
            cv_caption = "No objects detected."
        else:
            cv_caption = ", ".join([f"{count} {name}(s)" for name, count in counts.items()])

        return motion_detected, cv_caption