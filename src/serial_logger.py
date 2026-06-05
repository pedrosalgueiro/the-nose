from __future__ import annotations

import argparse
import csv
import os
import time

import serial

FIELDNAMES = ["timestamp_ms", "voc", "pm25", "temp", "humidity", "label"]


def parse_line(line: str) -> dict | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        return {
            "timestamp_ms": int(float(parts[0])),
            "voc": float(parts[1]),
            "pm25": float(parts[2]),
            "temp": float(parts[3]),
            "humidity": float(parts[4]),
        }
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Grava dados dos sensores com uma etiqueta para treino.")
    parser.add_argument("--port", required=True, help="Ex.: COM3, /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--label", required=True, help="Ex.: cafe, perfume, ar_limpo")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output", default="data/readings.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)

    with serial.Serial(args.port, args.baud, timeout=2) as ser, open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        end_time = time.time() + args.seconds
        count = 0
        print(f"A gravar '{args.label}' durante {args.seconds}s...")
        while time.time() < end_time:
            raw = ser.readline().decode(errors="ignore").strip()
            row = parse_line(raw)
            if row is None:
                continue
            row["label"] = args.label
            writer.writerow(row)
            count += 1
            print(row)

    print(f"Concluído. {count} linhas gravadas em {args.output}")


if __name__ == "__main__":
    main()
