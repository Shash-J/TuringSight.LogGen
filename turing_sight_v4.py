import time
import json
import ssl
import paho.mqtt.client as mqtt
from datetime import datetime
import cv2
import threading

from vision_engine import VisionEngine
from vlm_handler import VLMHandler

# --- TIMING CONFIG ---
STATIONARY_HEARTBEAT_SEC = 10    # Log frequency when no motion (Heartbeat)
MOTION_VLM_COOLDOWN_SEC = 20     # How often to call VLM during constant motion
PERIODIC_VLM_CHECK_SEC = 180     # Force VLM every 3 mins even if no motion
LOOP_SPEED = 5                   # Main loop delay (seconds)

# --- AWS CONFIG ---
AWS_ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"
TOPIC = "edge/logs"
USER_ID = "user1"
DEVICE_ID = "office-edge-node-01"

# Global state
last_vlm_paragraph = "System started. Initializing observation..."
vlm_is_running = False

def vlm_thread_worker(vlm_engine, frame, cv_caption, reason):
    global last_vlm_paragraph, vlm_is_running
    try:
        print(f" [VLM_START] Processing {reason} reasoning...")
        start = time.time()
        result = vlm_engine.generate_paragraph(frame, cv_caption, reason)
        last_vlm_paragraph = result
        print(f" [VLM_FINISH] Took {time.time()-start:.1f}s. Result: {result[:50]}...")
    finally:
        vlm_is_running = False

def run_pipeline():
    global last_vlm_paragraph, vlm_is_running
    
    cv_engine = VisionEngine()
    vlm_engine = VLMHandler()
    cap = cv2.VideoCapture("rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0")
    
    # MQTT Setup
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.tls_set(ca_certs="AmazonRootCA1.pem", certfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt", keyfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key")
    mqtt_client.connect(AWS_ENDPOINT, 8883)
    mqtt_client.loop_start()

    last_motion_vlm_time = 0
    last_periodic_vlm_time = time.time()
    last_heartbeat_time = 0

    print(f"\n[{datetime.now()}] >>> TURING SIGHT V4: ONLINE")

    while True:
        for _ in range(15): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: 
            print(" [ERROR] Stream lost. Retrying...")
            time.sleep(2); continue

        # 1. Deterministic Analysis
        motion_detected, p_count, cv_caption = cv_engine.get_analysis(frame)
        now = time.time()
        
        trigger_vlm = False
        reason = ""

        # 2. LOGIC: Decide if we call the VLM (The "Thinking" part)
        if not vlm_is_running:
            # Case A: Periodic Forced Check
            if (now - last_periodic_vlm_time) > PERIODIC_VLM_CHECK_SEC:
                trigger_vlm, reason = True, "PERIODIC_CHECK"
                last_periodic_vlm_time = now
            
            # Case B: Motion Detected + Cooldown
            elif motion_detected and (now - last_motion_vlm_time) > MOTION_VLM_COOLDOWN_SEC:
                trigger_vlm, reason = True, "MOTION_DETECTED"
                last_motion_vlm_time = now

        # 3. Execute VLM in background
        if trigger_vlm:
            vlm_is_running = True
            threading.Thread(target=vlm_thread_worker, args=(vlm_engine, frame.copy(), cv_caption, reason), daemon=True).start()

        # 4. LOGIC: MQTT Publishing (The "Reporting" part)
        should_publish = False
        log_type = ""

        if motion_detected:
            should_publish = True
            log_type = "MOTION_LOG"
        elif (now - last_heartbeat_time) > STATIONARY_HEARTBEAT_SEC:
            should_publish = True
            log_type = "STATIONARY_HEARTBEAT"
            last_heartbeat_time = now

        if should_publish:
            payload = {
                "userid": USER_ID,
                "device": DEVICE_ID,
                "event": f"[{log_type}] {cv_caption}",
                "object": last_vlm_paragraph,
                "timestamp": int(now),
                "motion": motion_detected,
                "person_count": p_count
            }
            mqtt_client.publish(TOPIC, json.dumps(payload))
            print(f" [{datetime.now().strftime('%H:%M:%S')}] {log_type} Published. People: {p_count}")

        time.sleep(LOOP_SPEED)

if __name__ == "__main__":
    run_pipeline()