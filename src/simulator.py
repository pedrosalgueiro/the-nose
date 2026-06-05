from __future__ import annotations

import math
import random
import time
from typing import Iterator

import numpy as np

from config import SENSOR_COLUMNS

BASELINES = {
    "ar_limpo": {
        "voc": 100, "nox": 1, "mq135_voltage": 0.58, "mq3_voltage": 0.35,
        "sht40_temperature": 23.0, "sht40_humidity": 55,
        "bme688_temperature": 23.4, "bme688_humidity": 50,
        "bme688_pressure": 994, "bme688_gas_resistance": 16000,
        "bme688_iaq": 50, "bme688_co2_equivalent": 500, "bme688_breath_voc_equivalent": 0.5,
        "scd40_co2": 650, "scd40_temperature": 23.0, "scd40_humidity": 52,
    },
    "cafe": {"voc": 210, "nox": 1.5, "mq135_voltage": 0.72, "mq3_voltage": 0.40, "bme688_gas_resistance": 10000, "bme688_iaq": 80, "bme688_breath_voc_equivalent": 0.9},
    "perfume": {"voc": 450, "nox": 2.5, "mq135_voltage": 1.10, "mq3_voltage": 1.30, "bme688_gas_resistance": 6500, "bme688_iaq": 150, "bme688_breath_voc_equivalent": 2.8},
    "alcool": {"voc": 520, "nox": 2.0, "mq135_voltage": 1.00, "mq3_voltage": 1.80, "bme688_gas_resistance": 6000, "bme688_iaq": 170, "bme688_breath_voc_equivalent": 3.3},
    "vinagre": {"voc": 310, "nox": 2.5, "mq135_voltage": 0.95, "mq3_voltage": 0.55, "bme688_gas_resistance": 8000, "bme688_iaq": 120, "bme688_breath_voc_equivalent": 1.8},
    "particulas": {"voc": 160, "nox": 5.0, "mq135_voltage": 0.80, "mq3_voltage": 0.45, "bme688_gas_resistance": 12000, "bme688_iaq": 100, "bme688_breath_voc_equivalent": 1.1, "scd40_co2": 700},
}

NOISE = {
    "voc": 8, "nox": 0.4, "mq135_voltage": 0.015, "mq3_voltage": 0.015,
    "sht40_temperature": 0.08, "sht40_humidity": 0.5,
    "bme688_temperature": 0.08, "bme688_humidity": 0.5,
    "bme688_pressure": 0.5, "bme688_gas_resistance": 250,
    "bme688_iaq": 3, "bme688_co2_equivalent": 10, "bme688_breath_voc_equivalent": 0.05,
    "scd40_co2": 12, "scd40_temperature": 0.08, "scd40_humidity": 0.5,
}


def generate_sequence(label: str, n: int = 180, seed: int | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    if label not in BASELINES:
        raise ValueError(f"Etiqueta desconhecida: {label}")

    base = dict(BASELINES["ar_limpo"])
    target = dict(base)
    target.update(BASELINES[label])
    rows = []

    for i in range(n):
        # Simulates the same collection protocol used for real data:
        # 0-30 s baseline, 30-90 s exposure, 90-180 s recovery.
        if label == "ar_limpo":
            intensity = 0.04 * math.sin(i / 12)
        elif i < 30:
            intensity = 0.02 * math.sin(i / 10)
        elif i < 90:
            # Smooth response during exposure.
            intensity = 1 / (1 + math.exp(-(i - 45) / 8))
        else:
            # Partial recovery after removing the sample.
            intensity = max(0.20, math.exp(-(i - 90) / 55))

        row = {"timestamp_ms": i * 1000}
        for sensor in SENSOR_COLUMNS:
            b = base.get(sensor, 0.0)
            t = target.get(sensor, b)
            value = b + intensity * (t - b)
            row[sensor] = round(max(0, value + rng.normal(0, NOISE.get(sensor, 0.1))), 3)
        row["label"] = label
        rows.append(row)
    return rows


def live_simulator(label: str) -> Iterator[dict]:
    while True:
        seq = generate_sequence(label, n=180, seed=random.randint(0, 1_000_000))
        for row in seq:
            row = dict(row)
            row["timestamp_ms"] = int(time.time() * 1000)
            row.pop("label", None)
            yield row
