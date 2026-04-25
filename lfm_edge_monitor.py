import cv2
import torch
import json
import time
import sys
import paho.mqtt.client as mqtt
from datetime import datetime
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- CONFIGURATION ---
RTSP_URL = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"
MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

if not torch.cuda.is_available():
    print("!!! ERROR: GPU NOT DETECTED !!!")
    sys.exit(1)

DEVICE = "cuda"

# AWS IoT CONFIG
ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"
TOPIC = "edge/logs"
DEVICE_ID = "rtx3050-surveillance-node"
USER_ID = "user1"

class LFMSurveillanceNode:
    def __init__(self):
        print(f"[System] Loading {MODEL_ID} to GPU...")
        # Use dtype="bfloat16" for the LFM2.5 backbone 
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True, 
            dtype="bfloat16",
            device_map=DEVICE
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"[System] {MODEL_ID} active on {torch.cuda.get_device_name(0)}")

    def analyze_behavior(self, frame):
            # Native resolution handles 512x512 without distortion 
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Updated text prompt: Asks for colors/distinctions and a strict format for querying
            prompt_text = (
                "Analyse what each people are doing and thier actions."
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            # Tokenize=True ensures we get tensors for the GPU
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(DEVICE)
            
            # Applying optimized generation parameters 
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=200,      # Increased slightly to accommodate structured data
                    temperature=0.1,         
                    min_p=0.15,              
                    repetition_penalty=1.05  
                )
            
            # --- THE FIX FOR THE "USER/ASSISTANT" OUTPUT ---
            # Calculate the length of the input prompt in tokens
            input_length = inputs['input_ids'].shape[1]
            
            # Slice the outputs to only keep the newly generated tokens
            generated_tokens = outputs[0][input_length:]
            
            # Decode only the generated response
            clean_text = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
            
            return clean_text

def run_loop():
    node = LFMSurveillanceNode()
    
    # AWS MQTT Setup
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.tls_set(
        ca_certs="AmazonRootCA1.pem", 
        certfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt", 
        keyfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key"
    )
    mqtt_client.connect(ENDPOINT, 8883)
    mqtt_client.loop_start()

    cap = cv2.VideoCapture(RTSP_URL)
    frame_count = 0
    
    print("--- TuringSight LFM2.5 Node: ONLINE (Text-Analysis Mode) ---")

    while True:
        for _ in range(20): cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            print("[Warning] Stream lost. Reconnecting...")
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        frame_count += 1
        # Analyze roughly once every 2 seconds to keep GPU cool
        if frame_count % 60 == 0:
            try:
                # 1. Get raw text from model
                description = node.analyze_behavior(frame)
                
                # 2. Wrap the text in the required JSON structure manually
                payload = {
                    "userid": USER_ID,
                    "device": DEVICE_ID,
                    "event": description.strip(),
                    "object": "Human Activity",
                    "timestamp": int(time.time()),
                    "model": "LFM2.5-VL-450M"
                }
                
                # 3. Publish to AWS
                mqtt_client.publish(TOPIC, json.dumps(payload))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Log Sent: {payload['event'][:70]}...")

            except Exception as e:
                print(f"[Error] Inference failed: {e}")

if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\n[System] Shutting down...")