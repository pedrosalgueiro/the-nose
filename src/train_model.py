from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from config import RANDOM_STATE
from features import make_feature_table


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    return len(counts) > 1 and counts.min() >= 2


def train(input_path: str, output_path: str, test_size: float = 0.25) -> None:
    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError("O CSV precisa de uma coluna 'label'.")
    if "session_id" not in df.columns:
        if "trial" in df.columns:
            print("[AVISO] CSV sem 'session_id'; a usar 'trial' como identificador de sessão.")
            df["session_id"] = df["label"].astype(str) + "_" + df["trial"].astype(str)
        else:
            print("[AVISO] CSV sem 'session_id'; cada label será tratado como uma única sessão. Para dados reais, usa o collect script.")
            df["session_id"] = df["label"].astype(str)
    if "elapsed_s" not in df.columns:
        print("[AVISO] CSV sem 'elapsed_s'; a derivar tempo relativo por ordem das linhas dentro de cada sessão.")
        df["elapsed_s"] = df.groupby("session_id").cumcount().astype(float)

    print("\nSessões disponíveis por classe:\n")
    print(df.groupby("label")["session_id"].nunique().sort_index())

    X, y = make_feature_table(df)
    if X.empty or y is None or y.empty:
        raise ValueError("Não há dados suficientes para criar features de treino.")

    print(f"\nExemplos de treino criados: {len(X)} sessões")
    print(f"Features por sessão: {len(X.columns)}")

    stratify = y if _can_stratify(y) else None
    if stratify is None:
        print("\n[AVISO] Poucas sessões por classe para split estratificado. A usar split simples.")

    # If the dataset is still tiny, do not make the test split too aggressive.
    if len(X) < 8:
        raise ValueError("Recolhe mais sessões antes de treinar: idealmente >= 8 sessões no total e >= 2 por classe.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=1,
        )),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    labels_sorted = sorted(y.unique())
    print("\nRelatório de classificação:\n")
    print(classification_report(y_test, preds, labels=labels_sorted, zero_division=0))
    print("Matriz de confusão, labels =", labels_sorted, "\n")
    print(confusion_matrix(y_test, preds, labels=labels_sorted))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "labels": labels_sorted,
        "training_mode": "segmented_session_features",
        "protocol": {
            "baseline_s": [0, 30],
            "exposure_s": [30, 90],
            "recovery_s": [90, 180],
        },
    }
    joblib.dump(bundle, output_path)
    print(f"\nModelo guardado em: {output_path}")
    print("Modo de treino: features segmentadas por sessão baseline/exposição/recuperação")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelo com sessões reais baseline→exposição→recuperação.")
    parser.add_argument("--input", default="data/real_readings.csv")
    parser.add_argument("--output", default="models/air_ai_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    train(args.input, args.output, args.test_size)


if __name__ == "__main__":
    main()
