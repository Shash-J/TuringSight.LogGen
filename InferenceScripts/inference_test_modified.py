import cv2
import base64
import requests
import time
from datetime import datetime

# --- CONFIG ---
RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# FIX 1: Downscaled resolution. 
# 1280x720 is still plenty of detail for a VLM to detect people and actions, 
# but will severely cut down on GPU heat and inference time.
FRAME_SIZE = (1280, 720)

def encode_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def run_turingsight():
    history = "Session started. Observe the surroundings."
    frame_count = 0
    
    cap = cv2.VideoCapture(RTSP_URL)
    print(f"Connected to: {RTSP_URL}")
    print("--- TuringSight: Active (Synchronous Logic) ---")

    while True:
        for _ in range(25): 
            cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Stream lost. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        frame_count += 1
        img_b64 = encode_frame(frame)
        timestamp = datetime.now().strftime("%H:%M:%S")

        if frame_count % 30 == 0:
            prompt_text = "Analyze this whole frame in general. Describe the overall scene, the environment, and any notable activities happening. Provide a comprehensive summary of about 100 to 200 words."
            analysis_type = "GENERAL SCENE ANALYSIS"
        else:
            prompt_text = "Analyze this frame. Focus strictly on the persons present and describe in detail the specific actions performed by them. Provide a clear summary of about 100 to 200 words."
            analysis_type = "PERSON & ACTION ANALYSIS"

        prompt = f"""
        [CONTEXT] {history}
        [TASK] {prompt_text}
        """

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.6,
                # FIX 2: Tripled the token limit to allow room for internal reasoning AND the summary
                "num_predict": 1500  
            }
        }

        try:
            print(f"\n[{timestamp}] Frame {frame_count} - Starting {analysis_type}...")
            start_time = time.time()
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status() 
            result = response.json()
            
            # FIX 3: Restored extraction logic to catch "thinking" vs "response" 
            # and added a debug dump so it's never completely blank.
            ai_summary = result.get("response", "").strip()
            thinking_log = result.get("thinking", "")
            
            if ai_summary:
                history = f"Previous observation snippet: {ai_summary[-200:]}"
            elif thinking_log:
                history = f"Previous thought snippet: {thinking_log[-200:]}"
            
            display_text = ai_summary
            if not display_text and thinking_log:
                display_text = "[Thinking Snippet (Cut off)]: " + thinking_log[-300:].replace('\n', ' ')
            elif not display_text and not thinking_log:
                display_text = f"[DEBUG] Raw API Output: {result}"

            elapsed = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) OUTPUT:")
            print(f">>> {display_text}")
            print("-" * 80)

        except requests.exceptions.Timeout:
            print(f"\n[{timestamp}] Error: Inference timed out.")
        except Exception as e:
            print(f"\n[{timestamp}] Error: {e}")

if __name__ == "__main__":
    run_turingsight()