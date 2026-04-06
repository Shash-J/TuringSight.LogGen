import requests
import base64
import json
import re

class VLMHandler:
    def __init__(self, model_name="qwen3-vl:4b", url="http://localhost:11434/api/generate"):
        self.url = url
        self.model = model_name

    def generate_paragraph(self, frame, cv_event_text):
        _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (1024, 576)))
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""
        [CV DATA] The computer vision system detected: {cv_event_text}.
        [TASK] Write a detailed 3-4 sentence paragraph describing the actions, 
        interactions, and scene context. Do not use JSON, just plain text.
        """
        
        payload = {"model": self.model, "prompt": prompt, "images": [img_b64], "stream": False}
        
        try:
            res = requests.post(self.url, json=payload, timeout=45)
            return res.json().get("response", "").strip()
        except:
            return "VLM Inference Failed."