import ssl
import json
import time
import paho.mqtt.client as mqtt

ENDPOINT = "a3dde9l2eto48i-ats.iot.ap-south-1.amazonaws.com"
PORT = 8883
TOPIC = "edge/logs"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code:", rc)

client = mqtt.Client()

client.on_connect = on_connect

client.tls_set(
    ca_certs="AmazonRootCA1.pem",
    certfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-certificate.pem.crt",
    keyfile="ee56753fba666c8a81345ef6875eed9a9b05d2caa5d681ccb5e239c7f7e156d0-private.pem.key",
    tls_version=ssl.PROTOCOL_TLSv1_2
)

client.connect(ENDPOINT, PORT)
client.loop_start()

time.sleep(2)   # wait for connection
i = 1
while i > 0:
    log = {
        "user_id": "user1",
        "device": "edge_node_01_cam_entrance(device_id)",
        "event": "new" + str(i) + " " + str(time.time()) + "log is to test the complete pipeline of edge->IoT core->dynamoDB and to Qdrant storage on collections",
        "object": "an event",
        "timestamp": int(time.time())
    }

    client.publish(TOPIC, json.dumps(log))
    print("Log sent" + str(i))
    i += 1

    time.sleep(10)