import requests
import base64
import cv2

class VLMHandler:
    def __init__(self, model_name="qwen3-vl:2b", url="http://localhost:11434/api/generate"):
        self.url = url
        self.model = model_name

    def generate_paragraph(self, frame, cv_caption, trigger_reason):
        _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (1024, 576)))
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""
        [CONTEXT] Office Surveillance. Trigger: {trigger_reason}.
        [DETECTION DATA] The CV engine sees: {cv_caption}.
        [TASK] Describe the activity and engagement of these people. 
        Are they working, standing, eating or using devices? 
        Write a small paragraph. No JSON.
        """
        
        payload = {"model": self.model, "prompt": prompt, "images": [img_b64], "stream": False}
        
        try:
            res = requests.post(self.url, json=payload, timeout=60)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            return f"VLM Analysis unavailable: {str(e)}"