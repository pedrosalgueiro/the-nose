from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from config import LABELS
from simulator import generate_sequence


def main() -> None:
    os.makedirs("data", exist_ok=True)
    all_rows = []
    for label in LABELS:
        for trial in range(18):
            session_id = f"{label}_{trial:03d}"
            rows = generate_sequence(label, n=180, seed=trial + 1000 * LABELS.index(label))
            for i, r in enumerate(rows):
                r["trial"] = trial
                r["session_id"] = session_id
                r["elapsed_s"] = float(i)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out = "data/demo_readings.csv"
    df.to_csv(out, index=False)
    print(f"Dataset criado: {out} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
