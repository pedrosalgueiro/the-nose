from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import paho.mqtt.client as mqtt

#from config import DEFAULT_MQTT_TOPICS, SENSOR_COLUMNS

try:
    from .config import DEFAULT_MQTT_TOPICS, SENSOR_COLUMNS
except ImportError:
    from config import DEFAULT_MQTT_TOPICS, SENSOR_COLUMNS
    

def _parse_float(payload: bytes) -> Optional[float]:
    text = payload.decode(errors="ignore").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in (
                "value", "state", "voc", "voc_index", "nox", "co2",
                "pm25", "temperature", "humidity", "pressure", "voltage",
                "gas_resistance", "iaq",
            ):
                if key in obj:
                    return float(obj[key])
    except Exception:
        return None
    return None


@dataclass
class MQTTConfig:
    broker: str
    port: int = 1883
    username: str = ""
    password: str = ""
    client_id: str = "air-ai-dashboard"
    topics: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MQTT_TOPICS))
    debug_messages: bool = False
    debug_payload: bool = False


class MQTTLiveReader:
    def __init__(self, cfg: MQTTConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest: Dict[str, float] = {
            "voc": 100.0,
            "nox": 1.0,
            "mq135_voltage": 0.0,
            "mq3_voltage": 0.0,
            "sht40_temperature": 22.0,
            "sht40_humidity": 50.0,
            "bme688_temperature": 22.0,
            "bme688_humidity": 50.0,
            "bme688_pressure": 1013.0,
            "bme688_gas_resistance": 0.0,
            "bme688_iaq": 50.0,
            "bme688_co2_equivalent": 500.0,
            "bme688_breath_voc_equivalent": 0.5,
            "scd40_co2": 0.0,
            "scd40_temperature": 22.0,
            "scd40_humidity": 50.0,
        }
        for col in SENSOR_COLUMNS:
            self._latest.setdefault(col, 0.0)

        self._last_seen: Dict[str, float] = {}
        self._connected = False
        self._error = ""
        self._topic_to_key = {topic: key for key, topic in cfg.topics.items() if topic}

        self.client = mqtt.Client(client_id=cfg.client_id, protocol=mqtt.MQTTv311)
        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password or None)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = True
        self._error = ""
        if self.cfg.debug_messages:
            print(f"[MQTT] Ligado ao broker {self.cfg.broker}:{self.cfg.port}")
        for topic, key in self._topic_to_key.items():
            client.subscribe(topic, qos=0)
            if self.cfg.debug_messages:
                print(f"[MQTT SUB] {key:32s} <- {topic}")

    def _on_disconnect(self, client, userdata, reason_code, properties=None):
        self._connected = False

    def _on_message(self, client, userdata, msg):
        key = self._topic_to_key.get(msg.topic)
        value = _parse_float(msg.payload)
        retain_flag = "RETAIN" if getattr(msg, "retain", False) else "LIVE"

        if self.cfg.debug_messages:
            payload_text = msg.payload.decode(errors="ignore").strip()
            if self.cfg.debug_payload:
                print(f"[MQTT RX {retain_flag}] {msg.topic} -> {key or '?'} | raw={payload_text!r} | parsed={value}")
            else:
                print(f"[MQTT RX {retain_flag}] {msg.topic} -> {key or '?'} = {value}")

        if key is None or value is None:
            return
        with self._lock:
            self._latest[key] = value
            self._last_seen[key] = time.time()

    def start(self):
        try:
            self.client.connect(self.cfg.broker, self.cfg.port, keepalive=30)
            self.client.loop_start()
        except Exception as exc:
            self._error = str(exc)
            raise

    def stop(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def read(self) -> dict:
        with self._lock:
            row = dict(self._latest)
            last_seen = dict(self._last_seen)
        row["timestamp_ms"] = int(time.time() * 1000)
        row["mqtt_connected"] = self._connected
        row["mqtt_last_seen"] = last_seen
        row["mqtt_error"] = self._error
        return row
