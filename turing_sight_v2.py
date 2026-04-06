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
MODEL_NAME = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOPIC = "edge/logs"
DEVICE_ID = "laptop-edge-node_cam-01"
USER_ID = "user1"

# Timing
MOTION_COOLDOWN = 5  # During motion, analyze every 5 seconds
HEARTBEAT_INTERVAL = 30 # If no motion, send a log every 30 seconds anyway

# AWS Certificates
CA_CERT = "AmazonRootCA1.pem"
CLIENT_CERT = "ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt"
PRIVATE_KEY = "ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key"
ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"

def setup_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.tls_set(ca_certs=CA_CERT, certfile=CLIENT_CERT, keyfile=PRIVATE_KEY, tls_version=ssl.PROTOCOL_TLSv1_2)
    client.connect(ENDPOINT, 8883)
    client.loop_start()
    return client

def get_vlm_analysis(frame, last_event):
    _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (1280, 720)))
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    prompt = f"""
    [CONTEXT] Previous state: {last_event}
    [TASK] Analyze this frame. Output strictly JSON with keys 'event' and 'object'. 
    Describe current actions/people.
    """
    
    payload = {"model": MODEL_NAME, "prompt": prompt, "images": [img_b64], "stream": False, "options": {"temperature": 0.2}}
    
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=45)
        raw_output = res.json().get("response", "")
        # Basic JSON extraction
        import re
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except:
        return None

def run_pipeline():
    mqtt_client = setup_mqtt()
    detector = MotionDetector(threshold=25, min_area=6000)
    cap = cv2.VideoCapture(RTSP_URL)
    
    last_vlm_call_time = 0
    last_heartbeat_time = 0
    last_known_event = "No previous activity recorded."
    
    print(">>> TuringSight Pipeline Active (Motion-Triggered Mode)")

    while True:
        # 1. Get Frame (Clear buffer for RTSP)
        for _ in range(5): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: break

        current_time = time.time()
        motion_detected = detector.has_motion(frame)

        # 2. Logic Branching
        should_call_vlm = False
        status_msg = ""

        if motion_detected:
            # Check if cooldown has passed
            if (current_time - last_vlm_call_time) > MOTION_COOLDOWN:
                should_call_vlm = True
                status_msg = "MOTION DETECTED - Triggering VLM"
        else:
            # If no motion for a long time, send a heartbeat log
            if (current_time - last_heartbeat_time) > HEARTBEAT_INTERVAL:
                status_msg = "NO CHANGE - Sending Periodic Heartbeat"
                should_call_vlm = False
                # Publish the previous state with new timestamp
                final_log = {
                    "userid": USER_ID, "device": DEVICE_ID,
                    "event": f"Stability Monitor: {last_known_event}",
                    "object": "Stationary Scene", "timestamp": int(current_time)
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
                final_log = {
                    "userid": USER_ID, "device": DEVICE_ID,
                    "event": last_known_event,
                    "object": vlm_data.get('object', 'Unknown'),
                    "timestamp": int(current_time)
                }
                mqtt_client.publish(TOPIC, json.dumps(final_log))
                last_vlm_call_time = current_time
                last_heartbeat_time = current_time # Reset heartbeat on actual event
                print(f"    - SUCCESS: Log published to AWS.")

        time.sleep(0.1)

if __name__ == "__main__":
    run_pipeline()