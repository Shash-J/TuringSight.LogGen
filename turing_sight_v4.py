import time
import json
import ssl
import paho.mqtt.client as mqtt
from datetime import datetime
import cv2

from vision_engine import VisionEngine
from vlm_handler import VLMHandler

# --- CONFIG ---
RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
AWS_ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"
TOPIC = "edge/logs"
COOLDOWN = 10 # Only call VLM every 10 seconds during motion

def run_production_pipeline():
    # Initialize components
    cv_engine = VisionEngine()
    vlm_engine = VLMHandler()
    cap = cv2.VideoCapture(RTSP_URL)
    
    # Setup MQTT
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.tls_set(ca_certs="AmazonRootCA1.pem", 
                       certfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt", 
                       keyfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key")
    mqtt_client.connect(AWS_ENDPOINT, 8883)
    mqtt_client.loop_start()

    last_vlm_time = 0
    last_vlm_paragraph = "System initialized. No motion detected yet."

    print(">>> TuringSight Production Pipeline Started")

    while True:
        # 1. Continuous Capture
        for _ in range(10): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: break

        # 2. Continuous CV Analysis (Runs every loop)
        motion_detected, cv_caption = cv_engine.get_deterministic_analysis(frame)
        
        current_time = time.time()
        
        # 3. Decision Logic
        if motion_detected and (current_time - last_vlm_time > COOLDOWN):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Motion Detected -> Calling VLM")
            # Generate the detailed paragraph (Object)
            last_vlm_paragraph = vlm_engine.generate_paragraph(frame, cv_caption)
            last_vlm_time = current_time
        
        # 4. Construct Production Payload
        # event = CV Caption | object = VLM Paragraph
        payload = {
            "userid": "user1",
            "device": "laptop-edge-01",
            "event": cv_caption,          # From YOLO (Deterministic)
            "object": last_vlm_paragraph, # From Qwen-VL (Inference)
            "timestamp": int(current_time),
            "is_motion": motion_detected
        }

        # 5. Reliable Publishing
        mqtt_client.publish(TOPIC, json.dumps(payload))
        print(f"Log Published: {cv_caption[:30]}...")

        time.sleep(0.5) # Loop pacing

if __name__ == "__main__":
    run_production_pipeline()