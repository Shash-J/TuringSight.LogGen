"""
AWS IoT Core MQTT publisher.

Publishes semantic logs to AWS IoT Core via MQTT with TLS mutual auth.
Certificate paths and endpoint are configured in configs/pipeline.yaml
and can be overridden via environment variables.
"""

import ssl
import json
import time
import os

import paho.mqtt.client as mqtt


class MQTTPublisher:
    """Publishes log records to AWS IoT Core over MQTT."""

    def __init__(self, cfg_mqtt: dict):
        self.endpoint = os.environ.get("AWS_IOT_ENDPOINT") or cfg_mqtt["endpoint"]
        self.port = cfg_mqtt.get("port", 8883)
        self.topic = os.environ.get("AWS_IOT_TOPIC") or cfg_mqtt["topic"]
        self.enabled = cfg_mqtt.get("enabled", False)

        # Certificate paths (relative to /app inside container)
        self.ca_cert = cfg_mqtt["ca_cert"]
        self.client_cert = cfg_mqtt["client_cert"]
        self.client_key = cfg_mqtt["client_key"]

        self.client = None
        self._connected = False

    def connect(self):
        """Establish TLS connection to AWS IoT Core."""
        if not self.enabled:
            print("[MQTT] Disabled in config — skipping connection.")
            return

        # Verify cert files exist
        for path, name in [
            (self.ca_cert, "CA cert"),
            (self.client_cert, "Client cert"),
            (self.client_key, "Client key"),
        ]:
            if not os.path.exists(path):
                print(f"[MQTT] WARNING: {name} not found at {path} — MQTT disabled.")
                self.enabled = False
                return

        self.client = mqtt.Client()

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self._connected = True
                print(f"[MQTT] Connected to {self.endpoint}")
            else:
                print(f"[MQTT] Connection failed with code {rc}")

        def on_disconnect(client, userdata, rc):
            self._connected = False
            print(f"[MQTT] Disconnected (rc={rc})")

        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect

        self.client.tls_set(
            ca_certs=self.ca_cert,
            certfile=self.client_cert,
            keyfile=self.client_key,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

        try:
            self.client.connect(self.endpoint, self.port)
            self.client.loop_start()
            # Wait briefly for connection
            time.sleep(2)
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            self.enabled = False

    def publish(self, log_record: dict):
        """Publish a log record to AWS IoT Core."""
        if not self.enabled or not self._connected:
            return

        try:
            payload = json.dumps(log_record, ensure_ascii=False)
            result = self.client.publish(self.topic, payload)
            if result.rc == 0:
                print(f"[MQTT] Log published to {self.topic}")
            else:
                print(f"[MQTT] Publish failed (rc={result.rc})")
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")

    def disconnect(self):
        """Gracefully disconnect."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("[MQTT] Disconnected.")
