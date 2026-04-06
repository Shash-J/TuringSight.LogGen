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
INTERVAL = 5 
FRAME_SIZE = (320, 320)

def encode_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def run_turingsight():
    # The starting history
    history = "Start of session."
    
    cap = cv2.VideoCapture(RTSP_URL)
    print(f"Connected to: {RTSP_URL}")

    while True:
        # 1. Grab Fresh Frame
        for _ in range(15): cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        img_b64 = encode_frame(frame)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 2. RAW PROMPT (No instructions, just data)
        # We just hand over the history and the image.
        prompt = f"History: {history}\nImage: [Current Video Frame]"

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.5} # Higher temp to prevent "stuck" output
        }

        try:
            start_time = time.time()
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            result = response.json()
            
            # 3. EXTRACT EVERYTHING
            thinking = result.get("thinking", "")
            ai_response = result.get("response", "").strip()
            
            # Update history for the next loop
            if ai_response:
                history = ai_response

            # 4. UNFILTERED TERMINAL OUTPUT
            print("\n" + "="*60)
            print(f"[{timestamp}] RAW AI OUTPUT (Took {time.time()-start_time:.1f}s)")
            print("="*60)
            
            if thinking:
                print(f"AI THINKING LOG:\n{thinking}\n")
                print("-" * 30)
            
            print(f"AI RESPONSE: {ai_response}")
            print("="*60)

        except Exception as e:
            print(f"\nError: {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_turingsight()