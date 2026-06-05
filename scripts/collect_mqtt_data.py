#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import DEFAULT_MQTT_TOPICS, SENSOR_COLUMNS
from mqtt_listener import MQTTConfig, MQTTLiveReader

FIELDNAMES = ["timestamp", "elapsed_s", *SENSOR_COLUMNS, "label", "session_id"]


class Collector:
    def __init__(self):
        self.running = True

    def stop(self, *_):
        self.running = False


def _parse_required(value: str) -> list[str]:
    value = (value or "all").strip()
    if value.lower() in {"all", "*"}:
        return list(SENSOR_COLUMNS)
    if value.lower() in {"none", ""}:
        return []
    required = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in required if item not in SENSOR_COLUMNS]
    if unknown:
        raise SystemExit(
            "Sensores desconhecidos em --required: "
            + ", ".join(unknown)
            + "\nSensores válidos: "
            + ", ".join(SENSOR_COLUMNS)
        )
    return required


def _print_topic_mapping(topics: dict[str, str], required: list[str]):
    print()
    print("[MQTT] Tópicos configurados:")
    required_set = set(required)
    for key, topic in topics.items():
        marker = "*" if key in required_set else " "
        print(f"  {marker} {key:32s} <- {topic}")
    print("  * = sensor obrigatório para iniciar a recolha")


def _fresh_keys(reader: MQTTLiveReader, required: Iterable[str], since: float, max_age: float | None = None) -> tuple[list[str], list[str]]:
    row = reader.read()
    last_seen = row.get("mqtt_last_seen", {}) or {}
    now = time.time()
    fresh = []
    missing = []
    for key in required:
        ts = last_seen.get(key)
        if ts is None:
            missing.append(key)
            continue
        if ts < since:
            missing.append(key)
            continue
        if max_age is not None and now - ts > max_age:
            missing.append(key)
            continue
        fresh.append(key)
    return fresh, missing


def _wait_for_fresh_data(
    reader: MQTTLiveReader,
    required: list[str],
    ignore_seconds: float,
    timeout: float,
    poll_interval: float = 0.25,
) -> float:
    """
    Ignore retained MQTT values immediately delivered after subscription, then wait until
    every required sensor has published at least one value after the ignore window.

    Returns the timestamp after which data is considered fresh.
    """
    print()
    print(f"[MQTT] A ignorar possíveis mensagens retidas durante {ignore_seconds:.1f}s...")
    time.sleep(max(0.0, ignore_seconds))
    fresh_since = time.time()

    if not required:
        print("[MQTT] --required=none: não vou esperar por sensores frescos.")
        return fresh_since

    print("[MQTT] A aguardar leituras frescas de:")
    for key in required:
        print(f"  - {key}")

    deadline = time.time() + timeout
    last_print = 0.0
    while time.time() < deadline:
        fresh, missing = _fresh_keys(reader, required, since=fresh_since)
        if not missing:
            print("[MQTT] Leituras frescas recebidas. A iniciar recolha.")
            return fresh_since

        now = time.time()
        if now - last_print > 3:
            print(
                f"[MQTT] Ainda à espera de {len(missing)} sensor(es): "
                + ", ".join(missing)
            )
            last_print = now
        time.sleep(poll_interval)

    _, missing = _fresh_keys(reader, required, since=fresh_since)
    raise SystemExit(
        "[ERRO] Timeout à espera de leituras frescas.\n"
        "Isto normalmente significa que o ESP32 está desligado, os tópicos estão errados, "
        "ou algum sensor obrigatório não está a publicar.\n"
        "Sensores em falta: "
        + ", ".join(missing)
        + "\n\nDicas:\n"
        "  - confirma que o ESP32 está online;\n"
        "  - confirma os tópicos MQTT;\n"
        "  - usa --required para exigir só os sensores que queres usar;\n"
        "  - exemplo: --required voc,nox,mq135_voltage,mq3_voltage,sht40_temperature,sht40_humidity\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Recolher dados MQTT do electronic-nose ESPHome para treino.")
    parser.add_argument("--label", required=True, help="Classe. Ex: ar_limpo, cafe, perfume, alcool, vinagre")
    parser.add_argument("--session", required=True, help="ID único. Ex: cafe_001")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--output", default="data/real_readings.csv")
    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "10.0.0.2"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--username", default=os.getenv("MQTT_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("MQTT_PASSWORD", ""))
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--debug-mqtt",
        action="store_true",
        help="Mostra no terminal cada mensagem MQTT recebida, incluindo tópico, sensor e valor.",
    )
    parser.add_argument(
        "--debug-mqtt-payload",
        action="store_true",
        help="Com --debug-mqtt, mostra também o payload MQTT bruto.",
    )

    # Proteção contra valores MQTT retidos/stale.
    parser.add_argument(
        "--ignore-retained-seconds",
        type=float,
        default=3.0,
        help="Segundos iniciais a ignorar após subscrever, para descartar mensagens MQTT retidas.",
    )
    parser.add_argument(
        "--fresh-timeout",
        type=float,
        default=75.0,
        help="Tempo máximo para esperar por leituras frescas dos sensores obrigatórios.",
    )
    parser.add_argument(
        "--required",
        default="all",
        help=(
            "Sensores que têm de publicar uma leitura nova antes da recolha começar. "
            "Usa 'all', 'none', ou uma lista separada por vírgulas. "
            "Ex: voc,nox,mq135_voltage,mq3_voltage,scd40_co2"
        ),
    )
    parser.add_argument(
        "--max-stale-seconds",
        type=float,
        default=45.0,
        help="Durante a recolha, aborta se algum sensor obrigatório ficar sem atualizar por mais do que este tempo.",
    )

    for key, topic in DEFAULT_MQTT_TOPICS.items():
        parser.add_argument(f"--topic-{key.replace('_', '-')}", default=os.getenv("MQTT_TOPIC_" + key.upper(), topic))
    args = parser.parse_args()

    topics = {}
    for key in SENSOR_COLUMNS:
        attr = "topic_" + key
        topics[key] = getattr(args, attr)

    required = _parse_required(args.required)
    _print_topic_mapping(topics, required)

    cfg = MQTTConfig(
        broker=args.broker,
        port=args.port,
        username=args.username,
        password=args.password,
        topics=topics,
        client_id=f"air-ai-collector-{os.getpid()}",
        debug_messages=args.debug_mqtt,
        debug_payload=args.debug_mqtt_payload,
    )
    reader = MQTTLiveReader(cfg)
    reader.start()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()

    ctl = Collector()
    signal.signal(signal.SIGINT, ctl.stop)
    signal.signal(signal.SIGTERM, ctl.stop)

    print("==============================================")
    print(" RECOLHA DE DADOS MQTT")
    print("==============================================")
    print(f"Label:      {args.label}")
    print(f"Session ID: {args.session}")
    print(f"Duração:    {args.duration}s")
    print(f"Output:     {output}")
    print(f"Broker:     {args.broker}:{args.port}")
    print("Protocolo sugerido: 0-30s ar limpo, 30-90s amostra, 90-180s recuperação")
    print("==============================================")

    try:
        fresh_since = _wait_for_fresh_data(
            reader=reader,
            required=required,
            ignore_seconds=args.ignore_retained_seconds,
            timeout=args.fresh_timeout,
        )

        start = time.time()
        next_sample = start
        with output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()

            while ctl.running:
                now = time.time()
                elapsed = now - start
                if elapsed >= args.duration:
                    break

                # If required sensors stop updating during collection, abort instead of writing stale data.
                if required:
                    _, stale = _fresh_keys(
                        reader,
                        required,
                        since=fresh_since,
                        max_age=args.max_stale_seconds,
                    )
                    if stale:
                        raise SystemExit(
                            "[ERRO] Um ou mais sensores obrigatórios deixaram de atualizar durante a recolha: "
                            + ", ".join(stale)
                        )

                if now >= next_sample:
                    row = reader.read()
                    out = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_s": round(elapsed, 3),
                        "label": args.label,
                        "session_id": args.session,
                    }
                    for col in SENSOR_COLUMNS:
                        out[col] = row.get(col, 0.0)
                    writer.writerow(out)
                    print(
                        f"{elapsed:6.1f}s | VOC={out['voc']} NOx={out['nox']} "
                        f"MQ135={out['mq135_voltage']} MQ3={out['mq3_voltage']} "
                        f"CO2={out['scd40_co2']} BME_IAQ={out['bme688_iaq']}"
                    )
                    next_sample += args.sample_interval
                time.sleep(0.05)
    finally:
        reader.stop()
        print(f"[OK] Dados guardados em {output}")


if __name__ == "__main__":
    main()
