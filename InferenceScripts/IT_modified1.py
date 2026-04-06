import cv2
import base64
import requests
import time
from datetime import datetime

# --- CONFIG ---
RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
FRAME_SIZE = (2560, 1440)

# HARD LIMITS
MAX_THINK_TIME = 45  # Max seconds the model is allowed to think before being cut off

def encode_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def reset_model():
    """Forces Ollama to unload the model from memory to clear bad states."""
    print("\n[System] Initiating Model Refresh...")
    try:
        # Setting keep_alive to 0 immediately unloads the model from the GPU
        requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "keep_alive": 0}, timeout=10)
        time.sleep(2)  # Give the VRAM a moment to flush
        print("[System] Model unloaded successfully. It will reload on the next frame.")
    except Exception as e:
        print(f"[System] Failed to refresh model: {e}")

def run_turingsight():
    history = "Session started. Observe the surroundings."
    frame_count = 0
    
    cap = cv2.VideoCapture(RTSP_URL)
    print(f"Connected to: {RTSP_URL}")
    print(f"--- TuringSight: Active (Sync + Fallback Logic | Timeout: {MAX_THINK_TIME}s) ---")

    while True:
        # Clear the RTSP buffer
        for _ in range(25): 
            cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Stream lost. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        frame_count += 1
        img_b64 = encode_frame(frame)
        timestamp = datetime.now().strftime("%H:%M:%S")

        if frame_count % 30 == 0:
            prompt_text = "Analyze this whole frame in general. Describe the overall scene and any notable activities. Provide a direct, final summary of 100 to 200 words. Do not output excessive internal reasoning."
            analysis_type = "GENERAL SCENE ANALYSIS"
        else:
            prompt_text = "Analyze this frame. Focus strictly on the persons present and describe their specific actions. Provide a direct, final summary of 100 to 200 words. Do not output excessive internal reasoning."
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
                "temperature": 0.4,   # Lowered to reduce rambling and improve stability
                "num_predict": 1024   # Capped to a reasonable limit for 200 words
            }
        }

        try:
            print(f"\n[{timestamp}] Frame {frame_count} - Starting {analysis_type}...")
            start_time = time.time()
            
            # Applying the hard timeout limit here
            response = requests.post(OLLAMA_URL, json=payload, timeout=MAX_THINK_TIME)
            response.raise_for_status() 
            result = response.json()
            
            full_output = result.get("response", "").strip()
            elapsed = time.time() - start_time
            
            # --- FALLBACK LOGIC: EMPTY RESPONSE ---
            if not full_output:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) OUTPUT ERROR: Empty string detected.")
                reset_model()
                history = "Session started. Observe the surroundings." # Reset context to prevent cascading errors
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) OUTPUT:")
                print(f">>>\n{full_output}\n<<<")
                history = f"Previous observation: {full_output[-200:]}"
                
        # --- FALLBACK LOGIC: TIMEOUT ---
        except requests.exceptions.Timeout:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ERROR: Model took longer than {MAX_THINK_TIME}s. Cutting process to save power.")
            reset_model()
            history = "Session started. Observe the surroundings."
            
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        print("-" * 80)
        print("[System] Waiting for 0.1 seconds before capturing the next frame...")
        time.sleep(0.1)

if __name__ == "__main__":
    run_turingsight()


