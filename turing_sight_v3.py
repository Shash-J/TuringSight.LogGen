import cv2
import time
import json
import ssl
import requests
import base64
import paho.mqtt.client as mqtt
from datetime import datetime
from motion_logic import MotionDetector

# --- CONFIGURATION ---
RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
MODEL_NAME = "qwen3-vl:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOPIC = "edge/logs"
DEVICE_ID = "laptop-edge-node_cam-01"
USER_ID = "user1"

# Timing
MOTION_COOLDOWN = 5          # During motion, analyze every 5 seconds
HEARTBEAT_INTERVAL = 20      # If no motion, send a text log every 5 seconds anyway
STATIONARY_VLM_INTERVAL = 60 # If no motion, force a VLM check every 120 seconds

# AWS Certificates
CA_CERT = "AmazonRootCA1.pem"
CLIENT_CERT = "ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt"
PRIVATE_KEY = "ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key"
ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"

def setup_mqtt():
    print("[DEBUG] Setting up MQTT connection...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.tls_set(ca_certs=CA_CERT, certfile=CLIENT_CERT, keyfile=PRIVATE_KEY, tls_version=ssl.PROTOCOL_TLSv1_2)
    client.connect(ENDPOINT, 8883)
    client.loop_start()
    print("[DEBUG] MQTT connection established.")
    return client

def get_vlm_analysis(frame, last_event):
    print("    [DEBUG] Preparing frame for VLM inference...")
    _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (1280, 720)))
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    prompt = f"""
    [CONTEXT] Previous state: {last_event}
    [TASK] Analyze this frame. 
    Describe current actions/people.
    """
    
    payload = {"model": MODEL_NAME, "prompt": prompt, "images": [img_b64], "stream": False, "options": {"temperature": 0.2}}
    
    try:
        print("    [DEBUG] Sending request to Ollama...")
        start_time = time.time()
        res = requests.post(OLLAMA_URL, json=payload, timeout=45)
        print(f"    [DEBUG] Request returned status code: {res.status_code} in {time.time() - start_time:.2f}s")
        
        res.raise_for_status() # Raise an exception for bad status codes
        
        raw_output = res.json().get("response", "")
        print(f"    [DEBUG] Raw output from VLM: {raw_output.strip()}")
        
        # Basic JSON extraction
        import re
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
            print("    [DEBUG] Successfully parsed JSON from output.")
            return parsed_json
        else:
            print("    [DEBUG] ERROR: Failed to find valid JSON block in VLM output.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"    [DEBUG] ERROR: Network or API issue with Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"    [DEBUG] ERROR: JSON Decode Error: {e} - Could not parse the matched string.")
        return None
    except Exception as e:
        print(f"    [DEBUG] ERROR: Unexpected error during VLM inference: {e}")
        return None

def run_pipeline():
    mqtt_client = setup_mqtt()
    print("[DEBUG] Initializing Motion Detector...")
    detector = MotionDetector(threshold=25, min_area=6000)
    print(f"[DEBUG] Connecting to RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("[DEBUG] CRITICAL ERROR: Could not open RTSP stream.")
        return
        
    last_vlm_call_time = 0
    last_heartbeat_time = time.time()
    last_stationary_vlm_call_time = time.time()
    last_known_event = "No previous activity recorded."
    
    print(">>> TuringSight Pipeline Active (Motion-Triggered & Forced Periodic Mode)")

    while True:
        # 1. Get Frame (Clear buffer for RTSP)
        for _ in range(5): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] Failed to grab frame from stream.")
            time.sleep(1)
            continue

        current_time = time.time()
        motion_detected = detector.has_motion(frame)

        # 2. Logic Branching
        should_call_vlm = False
        is_stationary_vlm = False
        status_msg = ""

        if motion_detected:
            # RESET heartbeat timer so we don't send periodic heartbeats right after motion
            last_heartbeat_time = current_time 
            last_stationary_vlm_call_time = current_time

            # Check if cooldown has passed
            if (current_time - last_vlm_call_time) > MOTION_COOLDOWN:
                should_call_vlm = True
                status_msg = "MOTION DETECTED - Triggering VLM"
        else:
            # First check if the 1-minute stationary VLM interval has passed
            if (current_time - last_stationary_vlm_call_time) > STATIONARY_VLM_INTERVAL:
                should_call_vlm = True
                is_stationary_vlm = True
                status_msg = "STATIONARY SCENE - Forced Periodic VLM Check"
                
            # If not time for VLM, check if it's time for the periodic heartbeat
            elif (current_time - last_heartbeat_time) > HEARTBEAT_INTERVAL:
                status_msg = "NO CHANGE - Sending Periodic Heartbeat"
                final_log = {
                    "userid": USER_ID, "device": DEVICE_ID,
                    "event": f"Stability Monitor: {last_known_event}",
                    "object": "Stationary Scene", 
                    "timestamp": int(current_time)
                }
                mqtt_client.publish(TOPIC, json.dumps(final_log))
                last_heartbeat_time = current_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_msg}")

        # 3. VLM Inference
        if should_call_vlm:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_msg}")
            vlm_data = get_vlm_analysis(frame, last_known_event)
            
            if vlm_data:
                last_known_event = vlm_data.get('event', last_known_event)
                
                # Apply extra logic based on whether it was a forced stationary call
                if is_stationary_vlm:
                    detected_object = "Stationary Scene"
                    last_stationary_vlm_call_time = current_time
                else:
                    detected_object = vlm_data.get('object', 'Unknown')
                    last_vlm_call_time = current_time
                
                final_log = {
                    "userid": USER_ID, "device": DEVICE_ID,
                    "event": last_known_event,
                    "object": detected_object,
                    "timestamp": int(current_time)
                }
                mqtt_client.publish(TOPIC, json.dumps(final_log))
                
                # Reset timers after successful AWS publish
                last_heartbeat_time = current_time 
                print(f"    - SUCCESS: Log published to AWS.")
            else:
                print("    [DEBUG] FAILED to get valid data from VLM. Skipping AWS publish.")

        time.sleep(0.1)

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n[DEBUG] Pipeline stopped by user.")