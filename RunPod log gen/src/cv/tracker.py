import math
from typing import Dict, Any

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Please run: pip install ultralytics")
    YOLO = None


class CustomCVMemory:
    """
    Maintains a structured JSON state for every tracked entity.
    This acts as the 'custom memory layer' before triggering the VLM.
    """
    def __init__(self, stationary_threshold_px: int = 20):
        self.entities: Dict[int, Dict[str, Any]] = {}
        self.stationary_threshold_px = stationary_threshold_px

    def update_entity(self, entity_id: int, class_name: str, bbox: list, ts: float):
        """
        Updates the memory state for a specific tracked entity.
        bbox is [x1, y1, x2, y2]
        ts is the exact capture timestamp (float epoch)
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                "id": entity_id,
                "class": class_name,
                "first_seen": ts,
                "last_seen": ts,
                "bbox": bbox,
                "center": (cx, cy),
                "is_stationary": False,
                "stationary_since": None
            }
            return

        entity = self.entities[entity_id]
        prev_cx, prev_cy = entity["center"]
        
        # Calculate distance moved since last frame
        dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        
        # Check if stationary
        if dist < self.stationary_threshold_px:
            if not entity["is_stationary"]:
                entity["is_stationary"] = True
                entity["stationary_since"] = ts
        else:
            entity["is_stationary"] = False
            entity["stationary_since"] = None
            
        entity["last_seen"] = ts
        entity["bbox"] = bbox
        entity["center"] = (cx, cy)
        
    def get_state(self, entity_id: int) -> dict:
        return self.entities.get(entity_id, {})
        
    def cleanup_stale(self, current_ts: float, max_age_sec: float = 30.0):
        """Removes entities that haven't been seen recently"""
        stale_ids = [eid for eid, data in self.entities.items() 
                     if (current_ts - data["last_seen"]) > max_age_sec]
        for eid in stale_ids:
            del self.entities[eid]


class CVTracker:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.3):
        if YOLO is None:
            raise RuntimeError("ultralytics library is required for CVTracker.")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.memory = CustomCVMemory()
        print(f"[CVTracker] Initialized with model {model_path} and ByteTrack")
        
    def process_frame(self, frame, timestamp: float):
        """
        Runs YOLO tracking on the frame and updates memory.
        Returns the updated state of all entities currently in frame.
        """
        # run tracking (persist=True ensures IDs match across frames)
        # verbose=False to keep logs clean
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.conf_thresh, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        active_ids = []
        if len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes
            
            for box in boxes:
                if box.id is None:
                    continue # Not tracked securely yet
                    
                entity_id = int(box.id[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                self.memory.update_entity(entity_id, class_name, bbox, timestamp)
                active_ids.append(entity_id)
                
        # Clean up old entities that disappeared from frame
        self.memory.cleanup_stale(timestamp)
        
        # Return state of currently active entities for this exact frame
        return {eid: self.memory.get_state(eid) for eid in active_ids}
