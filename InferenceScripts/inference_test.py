import cv2
import base64
import requests
import json
import time
from datetime import datetime

# --- CONFIG ---
RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
INTERVAL = 10
FRAME_SIZE = (2560, 1440)

def encode_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def run_turingsight():
    # History starts as the initial state
    history = "Session started. Observe the surroundings."
    
    cap = cv2.VideoCapture(RTSP_URL)
    print(f"Connected to: {RTSP_URL}")
    print("--- TuringSight: Active (Fixed Logic) ---")

    while True:
        # 1. Grab Fresh Frame
        for _ in range(25): cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Stream lost. Reconnecting...")
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        img_b64 = encode_frame(frame)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 2. THE PROMPT
        # We explicitly ask for a final result after the reasoning.
        prompt = f"""
        [CONTEXT] {history}
        [TASK] Analyze this frame. Describe current activity. 
        End your response with a clear summary of the scene.
        """

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_predict": 500  # Increased to 500 to allow room for THINKING + SUMMARY
            }
        }

        try:
            start_time = time.time()
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            result = response.json()
            
            # 3. EXTRACT DATA
            thinking_log = result.get("thinking", "")
            ai_summary = result.get("response", "").strip()
            
            # 4. RECURSIVE LOGIC: Update history with the detailed thinking
            if thinking_log:
                history = thinking_log
            
            # 5. SMART TERMINAL OUTPUT
            # If the summary is empty, we show the end of the thinking log so the terminal isn't blank
            display_text = ai_summary
            if not display_text and thinking_log:
                # Take the last 200 characters of the thinking log as a fallback
                display_text = "[Thinking Snippet]: " + thinking_log[-200:].replace('\n', ' ')

            elapsed = time.time() - start_time
            print(f"\n[{timestamp}] ({elapsed:.1f}s) ANALYSIS UPDATE:")
            print(f">>> {display_text if display_text else 'AI is processing...'}")
            print("-" * 60)

        except Exception as e:
            print(f"\n[{timestamp}] Error: {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_turingsight()