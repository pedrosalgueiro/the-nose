import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import make_fast_feature_table


RANDOM_STATE = 42

EXCLUDED_SESSIONS = {
    "coffee_outside_1",
    "coffee_outside_2",
    "coffee_outside_3",
    "coffee_outside_4",
    "coffee_outside_5",
}


def train(input_path: str, output_path: str, test_size: float = 0.25):
    df = pd.read_csv(input_path)

    required = {"label", "session_id", "elapsed_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ignorar recovery no modelo rápido principal
    df = df[
        ~df["label"].str.contains("recover", case=False, na=False)
        & ~df["session_id"].str.contains("recover", case=False, na=False)
        & ~df["session_id"].str.contains("lemon", case=False, na=False)
    ].copy()

    # Excluir sessões com geometria antiga/fraca
    df = df[~df["session_id"].isin(EXCLUDED_SESSIONS)].copy()

    # Opcional: excluir lemon enquanto só houver poucas sessões
    counts = df.groupby("label")["session_id"].nunique()
    valid_labels = counts[counts >= 5].index
    df = df[df["label"].isin(valid_labels)].copy()

    # Garantir que cada sessão tem dados até pelo menos 60 s
    max_elapsed = df.groupby("session_id")["elapsed_s"].max()
    valid_sessions = max_elapsed[max_elapsed >= 60].index
    df = df[df["session_id"].isin(valid_sessions)].copy()

    X, y, session_ids = make_fast_feature_table(df)

    print()
    print("Fast model dataset:")
    print(f"Rows: {len(df):,}")
    print(f"Sessions: {len(X):,}")
    print(f"Features: {X.shape[1]:,}")
    print()
    print("Sessions per label:")
    print(y.value_counts().to_string())

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test, session_train, session_test = train_test_split(
        X,
        y,
        session_ids,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print()
    print("Classification report:")
    print(classification_report(y_test, preds))

    labels_sorted = sorted(y.unique())

    print()
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, preds, labels=labels_sorted),
            index=[f"true_{label}" for label in labels_sorted],
            columns=[f"pred_{label}" for label in labels_sorted],
        )
    )

    results = pd.DataFrame(
        {
            "session_id": session_test.values,
            "true_label": y_test.values,
            "predicted_label": preds,
        }
    )

    errors = results[results["true_label"] != results["predicted_label"]]

    print()
    print("Misclassified sessions:")
    if errors.empty:
        print("None")
    else:
        print(errors.to_string(index=False))

    rf = model.named_steps["clf"]

    importance_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": rf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print()
    print("Top 30 features:")
    print(importance_df.head(30).to_string(index=False))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "labels": labels_sorted,
        "training_mode": "fast_early_exposure",
        "protocol": {
            "baseline_s": [0, 25],
            "early_exposure_s": [35, 60],
        },
        "excluded_sessions": sorted(EXCLUDED_SESSIONS),
    }

    joblib.dump(bundle, output_path)

    importance_path = output_path.with_name(output_path.stem + "_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    print()
    print(f"Saved fast model to: {output_path}")
    print(f"Saved feature importance to: {importance_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/real_readings.csv")
    parser.add_argument("--output", default="models/electronic_nose_fast.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    train(args.input, args.output, args.test_size)


if __name__ == "__main__":
    main()
