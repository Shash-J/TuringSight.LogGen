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
# RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
RTSP_URL = "rtsp://716f898c7b71.entrypoint.cloud.wowza.com:1935/app-8F9K44lJ/304679fe_stream2"
MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
FRAME_SIZE = (1280, 720) # Downscaled for stable GPU inference
MAX_THINK_TIME = 45 

# --- AWS IoT & MQTT CONFIG ---
ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"
PORT = 8883
TOPIC = "edge/logs"
DEVICE_ID = "laptop-edge-node_cam-01"
USER_ID = "user1"

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
        # Search for the first '{' and last '}' to strip markdown like ```json
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None

def run_turingsight():
    history = "Session started. Observe the surroundings."
    frame_count = 0
    
    # Initialize MQTT
    print("[System] Connecting to AWS IoT...")
    mqtt_client = setup_mqtt()
    print(f"[System] Connected to endpoint: {ENDPOINT}")
    
    cap = cv2.VideoCapture(RTSP_URL)
    print(f"Connected to Camera: {RTSP_URL}")
    print(f"--- TuringSight Edge: Active (JSON Formatting | AWS Publish) ---")

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
        current_unix_time = int(time.time())

        # Strict JSON prompt formulation
        base_instruction = "You MUST output your response strictly as a JSON object with exactly two keys: 'event' and 'object'. Do NOT output markdown, backticks, or internal reasoning."
        
        if frame_count % 30 == 0:
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
                "temperature": 0.2,   # Extremely low temp to enforce strict JSON formatting
                "num_predict": 1024   
            }
        }

        try:
            print(f"\n[{timestamp}] Frame {frame_count} - Starting {analysis_type}...")
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
                continue

            # Attempt to parse the AI's output into a dictionary
            parsed_data = parse_llm_json(raw_output)
            
            if parsed_data and "event" in parsed_data:
                # Construct the final JSON payload
                final_log = {
                    "userid":USER_ID,
                    "device": DEVICE_ID,
                    "event": parsed_data.get("event", "No event description provided."),
                    "object": parsed_data.get("object", "Unknown"),
                    "timestamp": current_unix_time
                }
                
                # Publish to AWS
                mqtt_client.publish(TOPIC, json.dumps(final_log))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) SUCCESS: Log published to {TOPIC}")
                print(f">>> JSON Payload: {json.dumps(final_log, indent=2)}")
                
                # Update history using just the event description
                history = f"Previous observation: {final_log['event'][-200:]}"
                
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}s) PARSING ERROR: Model failed to return valid JSON.")
                print(f">>> Raw Output: {raw_output}")
                reset_model() # Reset to clear bad prompt formatting habits
                
        except requests.exceptions.Timeout:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ERROR: Model took longer than {MAX_THINK_TIME}s.")
            reset_model()
            history = "Session started. Observe the surroundings."
            
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        print("-" * 80)
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        run_turingsight()
    except KeyboardInterrupt:
        print("\n[System] Shutting down TuringSight...")