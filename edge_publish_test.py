import cv2
import base64
import requests
import time
import json
import ssl
import re
import paho.mqtt.client as mqtt
from datetime import datetime

# --- OLLAMA & VISION CONFIG ---
VIDEO_PATH = r"C:\Users\Shashanka\Desktop\TuringSight\upData database\raw_videos\pcvideo.mp4"   # <-- INSERT YOUR VIDEO PATH HERE
MODEL_NAME = "qwen3-vl:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
FRAME_SIZE = (1280, 720) 

# TIMING & LIMITS
VIDEO_SAMPLE_INTERVAL = 5  # Analyze 1 frame every 5 seconds of the video
MAX_THINK_TIME = 45        # Max seconds the model is allowed to think
GPU_REST_TIME = 1          # Seconds to let the GPU breathe between frames

# --- AWS IoT & MQTT CONFIG ---
ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"
PORT = 8883
TOPIC = "edge/logs"
DEVICE_ID = "laptop-edge-node_cam-01"

# AWS Certificates Paths (Ensure these are in the same directory as this script)
CA_CERT = "AmazonRootCA1.pem"
CLIENT_CERT = "ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt"
PRIVATE_KEY = "ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key"

def setup_mqtt():
    """Initializes and returns the configured MQTT client."""
    client = mqtt.Client()
    client.tls_set(
        ca_certs=CA_CERT,
        certfile=CLIENT_CERT,
        keyfile=PRIVATE_KEY,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.connect(ENDPOINT, PORT)
    client.loop_start()
    return client

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

def parse_llm_json(raw_text):
    """Extracts valid JSON from the LLM output, ignoring conversational filler/markdown."""
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None

def run_turingsight_video_to_aws():
    history = "Session started. Observe the surroundings."
    analysis_count = 0
    
    # 1. Initialize AWS MQTT
    print("[System] Connecting to AWS IoT...")
    try:
        mqtt_client = setup_mqtt()
        print(f"[System] Successfully connected to endpoint: {ENDPOINT}")
    except Exception as e:
        print(f"[Error] Failed to connect to AWS IoT Core: {e}")
        return

    # 2. Open Local Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Could not open video file at: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    frames_to_skip = int(fps * VIDEO_SAMPLE_INTERVAL)
    current_frame_pos = 0

    print(f"\nLoaded Video: {VIDEO_PATH}")
    print(f"Video Stats: {fps:.2f} FPS | Duration: {video_duration:.1f}s | Total Frames: {total_frames}")
    print(f"--- TuringSight Edge: Active (Local Video -> AWS IoT Core) ---")

    while True:
        # Jump directly to the specific timestamp in the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = cap.retrieve()
        
        if not ret:
            print("\n[System] End of video reached. Analysis complete.")
            break

        analysis_count += 1
        img_b64 = encode_frame(frame)
        current_unix_time = int(time.time())
        
        # Calculate video timestamp for your local logs
        video_seconds = current_frame_pos / fps
        m, s = divmod(int(video_seconds), 60)
        video_timestamp = f"{m:02d}:{s:02d}"

        # Strict JSON prompt formulation
        base_instruction = "You MUST output your response strictly as a JSON object with exactly two keys: 'event' and 'object'. Do NOT output markdown, backticks, or internal reasoning."
        
        if analysis_count % 10 == 0: 
            prompt_text = f"Analyze this whole frame. {base_instruction} \n- 'event': A general summary of the scene and activities (100-200 words).\n- 'object': A short phrase describing the main entities focused on."
            analysis_type = "GENERAL SCENE ANALYSIS"
        else:
            prompt_text = f"Analyze this frame. Focus strictly on the persons present. {base_instruction} \n- 'event': A summary of the persons and their specific actions (100-200 words).\n- 'object': A short phrase describing the specific people detected."
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
                "temperature": 0.2,   
                "num_predict": 1024   
            }
        }

        try:
            print(f"\n[Video Time: {video_timestamp}] Analysis #{analysis_count} - Starting {analysis_type}...")
            start_time = time.time()
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=MAX_THINK_TIME)
            response.raise_for_status() 
            result = response.json()
            
            raw_output = result.get("response", "").strip()
            elapsed = time.time() - start_time
            
            if not raw_output:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) OUTPUT ERROR: Empty string detected.")
                reset_model()
                history = "Session started. Observe the surroundings." 
            else:
                parsed_data = parse_llm_json(raw_output)
                
                if parsed_data and "event" in parsed_data:
                    # Construct the final JSON payload
                    final_log = {
                        "device": DEVICE_ID,
                        "event": parsed_data.get("event", "No event description provided."),
                        "object": parsed_data.get("object", "Unknown"),
                        "timestamp": current_unix_time
                    }
                    
                    # PUBLISH TO AWS
                    mqtt_client.publish(TOPIC, json.dumps(final_log))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) SUCCESS: Log published to AWS topic '{TOPIC}'")
                    print(f">>> AWS Payload: {json.dumps(final_log, indent=2)}")
                    
                    # Keep history concise by only retaining the event text
                    history = f"Previous observation: {final_log['event'][-200:]}"
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) PARSING ERROR: Model failed to return valid JSON.")
                    print(f">>> Raw Output: {raw_output}")
                    reset_model()
                
        except requests.exceptions.Timeout:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ERROR: Model took longer than {MAX_THINK_TIME}s.")
            reset_model()
            history = "Session started. Observe the surroundings."
            
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        print("-" * 80)
        
        # Advance the video and rest the GPU
        current_frame_pos += frames_to_skip
        time.sleep(GPU_REST_TIME)

    # Clean up
    cap.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("\n[System] TuringSight Edge shutdown sequence complete.")

if __name__ == "__main__":
    try:
        run_turingsight_video_to_aws()
    except KeyboardInterrupt:
        print("\n[System] Shutting down TuringSight...")