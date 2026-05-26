import json
import logging
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class MQTTPublisher:
    """
    MQTT Publisher for Home Assistant Auto-Discovery integration.
    Inspired by RuView's smart home integration.
    """
    def __init__(self, broker_url="127.0.0.1", port=1883):
        self.broker_url = broker_url
        self.port = port
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="AletheiaSensor")
        self.client.on_connect = self._on_connect
        self.connected = False
        self.last_prediction = None
        
        try:
            self.client.connect_async(self.broker_url, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.warning(f"[MQTT] Failed to initialize MQTT client: {e}")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info(f"[MQTT] Connected to Home Assistant Broker at {self.broker_url}")
            self.connected = True
            self._publish_ha_discovery()
        else:
            logger.warning(f"[MQTT] Connection failed with code {reason_code}")

    def _publish_ha_discovery(self):
        # Auto-Discovery for Activity State
        state_config = {
            "name": "Aletheia Room State",
            "state_topic": "aletheia/sensor/state",
            "unique_id": "aletheia_room_state_01",
            "device": {
                "identifiers": ["aletheia_v2"],
                "name": "Aletheia Wi-Fi Radar",
                "model": "BFI Passive Sensor",
                "manufacturer": "RuView/Aletheia Engine"
            }
        }
        self.client.publish("homeassistant/sensor/aletheia/state/config", json.dumps(state_config), retain=True)

        # Auto-Discovery for Fall Alert
        fall_config = {
            "name": "Aletheia Fall Alert",
            "state_topic": "aletheia/sensor/fall",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device_class": "safety",
            "unique_id": "aletheia_fall_alert_01",
            "device": {
                "identifiers": ["aletheia_v2"],
                "name": "Aletheia Wi-Fi Radar"
            }
        }
        self.client.publish("homeassistant/binary_sensor/aletheia/fall/config", json.dumps(fall_config), retain=True)

    def publish_state(self, prediction: str):
        if not self.connected or prediction == self.last_prediction:
            return
            
        self.last_prediction = prediction
        
        # Publish the raw state (EMPTY, PRESENCE, WALKING, FALLING, CALIBRATING)
        self.client.publish("aletheia/sensor/state", prediction)
        
        # Publish binary safety alert
        fall_status = "ON" if prediction == "FALLING" else "OFF"
        self.client.publish("aletheia/sensor/fall", fall_status)

# Singleton instance
mqtt_publisher = MQTTPublisher()
