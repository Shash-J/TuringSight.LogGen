import cv2
import base64
import requests
import time
from datetime import datetime

# --- CONFIG ---
# Placeholder for your local video file path
VIDEO_PATH = r"C:\Users\Shashanka\Desktop\TuringSight\upData database\raw_videos\pcvideo.mp4" 

MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
FRAME_SIZE = (1280, 720)

# TIMING & LIMITS
VIDEO_SAMPLE_INTERVAL = 2  # Analyze 1 frame every X seconds of the video
MAX_THINK_TIME = 45        # Max seconds the model is allowed to think
GPU_REST_TIME = 1          # Seconds to let the GPU breathe between frames

def encode_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def reset_model():
    """Forces Ollama to unload the model from memory to clear bad states."""
    print("\n[System] Initiating Model Refresh...")
    try:
        requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "keep_alive": 0}, timeout=10)
        time.sleep(2)  
        print("[System] Model unloaded successfully.")
    except Exception as e:
        print(f"[System] Failed to refresh model: {e}")

def run_turingsight_local():
    history = "Session started. Observe the surroundings."
    analysis_count = 0
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[Error] Could not open video file at: {VIDEO_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    # Calculate how many frames to skip to match the desired second interval
    frames_to_skip = int(fps * VIDEO_SAMPLE_INTERVAL)
    current_frame_pos = 0

    print(f"Loaded Video: {VIDEO_PATH}")
    print(f"Video Stats: {fps:.2f} FPS | Duration: {video_duration:.1f}s | Total Frames: {total_frames}")
    print(f"--- TuringSight: Active (Local Video Mode | Analyzing every {VIDEO_SAMPLE_INTERVAL}s) ---")

    while True:
        # Jump directly to the exact frame position in the video timeline
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = cap.retrieve()
        
        if not ret:
            print("\n[System] End of video reached. Analysis complete.")
            break

        analysis_count += 1
        img_b64 = encode_frame(frame)
        
        # Calculate the actual timestamp within the video (MM:SS)
        video_seconds = current_frame_pos / fps
        m, s = divmod(int(video_seconds), 60)
        video_timestamp = f"{m:02d}:{s:02d}"

        if analysis_count % 15 == 0:  # Adjusted to 15 since we are sampling slower
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
                "temperature": 0.4,   
                "num_predict": 1024   
            }
        }

        try:
            print(f"\n[Video Time: {video_timestamp}] Analysis #{analysis_count} - Starting {analysis_type}...")
            start_time = time.time()
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=MAX_THINK_TIME)
            response.raise_for_status() 
            result = response.json()
            
            full_output = result.get("response", "").strip()
            elapsed = time.time() - start_time
            
            # --- FALLBACK LOGIC ---
            if not full_output:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) OUTPUT ERROR: Empty string detected.")
                reset_model()
                history = "Session started. Observe the surroundings." 
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] (Processed in {elapsed:.1f}s) OUTPUT:")
                print(f">>>\n{full_output}\n<<<")
                history = f"Previous observation: {full_output[-200:]}"
                
        except requests.exceptions.Timeout:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ERROR: Model took longer than {MAX_THINK_TIME}s.")
            reset_model()
            history = "Session started. Observe the surroundings."
            
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        print("-" * 80)
        
        # Advance the target frame position for the next loop
        current_frame_pos += frames_to_skip
        
        print(f"[System] Resting GPU for {GPU_REST_TIME}s...")
        time.sleep(GPU_REST_TIME)

    cap.release()

if __name__ == "__main__":
    run_turingsight_local()