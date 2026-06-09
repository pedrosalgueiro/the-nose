


import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an electronic-nose dataset."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/real_readings.csv",
        help="Path to dataset CSV file. Default: data/real_readings.csv",
    )
    parser.add_argument(
        "--show-sessions",
        action="store_true",
        help="Show rows/duration for each individual session.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required = {"label", "session_id", "elapsed_s"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    print(f"File: {dataset_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Total sessions: {df['session_id'].nunique():,}")
    print(f"Total labels: {df['label'].nunique():,}")
    print()

    print("Rows per label:")
    print(
        df.groupby("label")
        .size()
        .sort_values(ascending=False)
        .rename("rows")
        .to_string()
    )
    print()

    print("Sessions per label:")
    print(
        df.groupby("label")["session_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("sessions")
        .to_string()
    )
    print()

    print("Rows/session statistics per label:")
    rows_per_session = (
        df.groupby(["label", "session_id"])
        .size()
        .rename("rows")
        .reset_index()
    )

    print(
        rows_per_session.groupby("label")["rows"]
        .agg(["count", "min", "mean", "max"])
        .round(1)
        .rename(columns={"count": "sessions"})
        .to_string()
    )
    print()

    print("Duration statistics per label:")
    durations = (
        df.groupby(["label", "session_id"])
        .agg(
            duration_s=("elapsed_s", lambda s: float(s.max() - s.min())),
            start_s=("elapsed_s", "min"),
            end_s=("elapsed_s", "max"),
        )
        .reset_index()
    )

    print(
        durations.groupby("label")["duration_s"]
        .agg(["count", "min", "mean", "max"])
        .round(1)
        .rename(columns={"count": "sessions"})
        .to_string()
    )

    if args.show_sessions:
        print()
        print("Rows per label and session:")
        summary = (
            df.groupby(["label", "session_id"])
            .agg(
                rows=("session_id", "size"),
                duration_s=("elapsed_s", lambda s: round(float(s.max() - s.min()), 1)),
                start_s=("elapsed_s", "min"),
                end_s=("elapsed_s", "max"),
            )
            .reset_index()
            .sort_values(["label", "session_id"])
        )

        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
