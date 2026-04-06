import cv2
import base64
import requests
import json
import time
from datetime import datetime

# --- CONFIG ---
RTSP_URL = "rtsp://admin:Admin123.@192.168.1.230:554/cam/realmonitor?channel=1&subtype=0"
MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
INTERVAL = 7  # Seconds between checks
FRAME_SIZE = (224, 224) # Smaller = Faster

def encode_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def run_turingsight():
    cap = cv2.VideoCapture(RTSP_URL)
    print(f"--- TuringSight: Light Mode Active ---")

    while True:
        # Clear buffer
        for _ in range(5): cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Stream lost. Retrying...")
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        img_b64 = encode_frame(frame)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # LIGHT PROMPT: Just asking for a simple sentence
        payload = {
            "model": MODEL_NAME,
            "prompt": "Describe this scene in one short sentence.",
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50 # Keep output very short
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                
                print(f"[{timestamp}] >> {summary}")
            else:
                print(f"[{timestamp}] Error: Ollama returned status {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"[{timestamp}] Error: Ollama is not running. Check your terminal.")
            time.sleep(5)
        except Exception as e:
            print(f"[{timestamp}] Error: {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_turingsight()