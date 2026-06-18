from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
try:
    import altair as alt
except Exception:
    alt = None
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import SENSOR_COLUMNS, DEFAULT_MQTT_TOPICS
    from src.features import features_for_latest_window, features_for_fast_window
    from src.mqtt_listener import MQTTConfig, MQTTLiveReader
except ImportError:
    from config import SENSOR_COLUMNS, DEFAULT_MQTT_TOPICS
    from features import features_for_latest_window, features_for_fast_window
    from mqtt_listener import MQTTConfig, MQTTLiveReader


st.set_page_config(page_title="Electronic Nose", page_icon="👃", layout="wide")


# -------------------------------------------------------------------
# Compact visual layout
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 0.8rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            max-width: 100vw;
        }

        div[data-testid="stMainBlockContainer"] {
            max-width: 100vw;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.8rem;
        }

        h1, h2, h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 0.65rem 0.8rem;
            border-radius: 0.75rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
        }

        .stAlert {
            padding-top: 0.45rem;
            padding-bottom: 0.45rem;
        }

        div[data-testid="stAlert"] {
            padding: 0.45rem 0.55rem;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stAlert"] p {
            font-size: 0.86rem;
            line-height: 1.18;
        }

        .compact-card {
            border: 1px solid #e5e7eb;
            border-radius: 0.9rem;
            padding: 0.85rem 1rem;
            background: #ffffff;
        }

        .small-caption {
            color: #64748b;
            font-size: 0.85rem;
        }

        section[data-testid="stSidebar"] {
            width: 300px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_FULL_MODEL_PATH = "models/electronic_nose_indoor_outdoor.joblib"
DEFAULT_FAST_MODEL_PATH = "models/electronic_nose_fast.joblib"

LOGO_PATH = "assets/electronic_nose_logo.png"

FULL_WINDOW_SIZE = 180
SESSION_END_S = 180.0

# Baseline used for the "ready for next sample" monitor.
# Use only the first 20 s to avoid accidentally including the moment
# when the sample starts approaching the nozzle.
READY_BASELINE_END_S = 20.0

RECOVERY_STABLE_SECONDS = 30.0

# Cleaner, less rigid "air is clean again" logic.
# Uses the continuous classifier instead of requiring every sensor to return
# exactly to the initial baseline.
CLEAN_AIR_READY_CONFIDENCE = 0.75
CLEAN_AIR_STABLE_SECONDS = 30.0
CLEAN_AIR_ALMOST_CONFIDENCE = 0.55

READY_THRESHOLDS = {
    "voc_abs_delta": 10.0,
    "mq135_abs_delta": 0.02,
    "mq3_abs_delta": 0.01,
    "bme688_iaq_abs_delta": 15.0,
    "bme688_bvoc_abs_delta": 1.0,
    "bme688_eco2_abs_delta": 80.0,
    "bme688_gas_relative_delta": 0.15,
}


@st.cache_resource
def load_model_bundle(path: str) -> dict[str, Any] | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Erro ao carregar modelo {path}: {exc}")
        return None


@st.cache_resource
def create_mqtt_reader(
    broker: str,
    port: int,
    username: str | None,
    password: str | None,
    topics_tuple: tuple[tuple[str, str], ...],
) -> MQTTLiveReader:
    cfg_kwargs = {
        "broker": broker,
        "port": port,
        "topics": dict(topics_tuple),
    }
    if username:
        cfg_kwargs["username"] = username
    if password:
        cfg_kwargs["password"] = password

    cfg = MQTTConfig(**cfg_kwargs)
    reader = MQTTLiveReader(cfg)
    reader.start()
    return reader


def normalize_window_records(records: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    """
    Take the latest window and re-index elapsed_s to 0..N-1.

    This is used for diagnostic/preview classifications where the original
    elapsed_s may be above 180 or where we want a partial session preview.
    """
    window = list(records[-window_size:])
    normalized: list[dict[str, Any]] = []

    for idx, row in enumerate(window):
        new_row = dict(row)
        new_row["elapsed_s"] = float(idx)
        normalized.append(new_row)

    return normalized


def resample_records_to_1hz(
    records: list[dict[str, Any]],
    *,
    duration_s: int = FULL_WINDOW_SIZE,
    pad_to_duration: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert irregular Streamlit readings into a regular 1 Hz session.

    Streamlit reruns can take longer than exactly 1 second, so a 180 s session
    may only contain 110-140 rows. The model was trained with ~1 row/s, so this
    function rebuilds a regular timeline using nearest/forward-filled readings.

    If pad_to_duration=True, missing future seconds are filled with the latest
    available reading. This is useful only for preview before 180 s.
    """

    if not records:
        return []

    df = pd.DataFrame(records).copy()

    if "elapsed_s" not in df.columns:
        df["elapsed_s"] = range(len(df))

    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df = df.dropna(subset=["elapsed_s"]).sort_values("elapsed_s")

    if df.empty:
        return []

    max_elapsed = float(df["elapsed_s"].max())

    if pad_to_duration:
        end_s = duration_s - 1
    else:
        end_s = min(duration_s - 1, int(max_elapsed))

    if end_s < 1:
        return []

    target = pd.DataFrame({"elapsed_s": [float(i) for i in range(end_s + 1)]})

    # Use merge_asof to map each target second to the most recent real reading.
    merged = pd.merge_asof(
        target,
        df,
        on="elapsed_s",
        direction="backward",
    )

    # Fill possible first rows if the first real reading happened after 0 s.
    merged = merged.ffill().bfill()

    # Force the target timeline after filling.
    merged["elapsed_s"] = target["elapsed_s"]

    return merged.to_dict(orient="records")




def rolling_records_last_seconds(records: list[dict[str, Any]], seconds: int) -> list[dict[str, Any]]:
    """
    Extract the last N seconds of monitor data and normalize elapsed_s to start at 0.

    Used for continuous rolling classification. This keeps the feature extractor
    aligned with the 0..180 s style windows used during training.
    """
    if not records:
        return []

    df = pd.DataFrame(records).copy()

    if "elapsed_s" not in df.columns:
        return normalize_window_records(records, seconds)

    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df = df.dropna(subset=["elapsed_s"]).sort_values("elapsed_s")

    if df.empty:
        return []

    max_t = float(df["elapsed_s"].max())
    df = df[df["elapsed_s"] >= max_t - seconds].copy()

    if df.empty:
        return []

    min_t = float(df["elapsed_s"].min())
    df["elapsed_s"] = df["elapsed_s"] - min_t

    return df.to_dict(orient="records")


def pad_session_to_full_window(records: list[dict[str, Any]], window_size: int = FULL_WINDOW_SIZE) -> list[dict[str, Any]]:
    """
    Build a 180-s shaped preview window before the session is complete.

    The full model was trained with a 180-s protocol. Before 180 s, this
    function pads the missing tail with the latest available reading. This
    allows a preview of how the final-model probabilities are evolving.

    This preview is intentionally not the official final prediction.
    """
    if not records:
        return []

    window = list(records[-window_size:])
    normalized: list[dict[str, Any]] = []

    for idx, row in enumerate(window):
        new_row = dict(row)
        new_row["elapsed_s"] = float(idx)
        normalized.append(new_row)

    while len(normalized) < window_size:
        last = dict(normalized[-1])
        last["elapsed_s"] = float(len(normalized))
        normalized.append(last)

    return normalized


def predict_from_records(
    records: list[dict[str, Any]],
    bundle: dict[str, Any] | None,
    mode: str,
    *,
    normalize_window: bool = False,
    pad_to_full_window: bool = False,
    regularize_1hz: bool = True,
) -> dict[str, Any] | None:
    if bundle is None:
        return None

    if regularize_1hz:
        if mode == "fast":
            records = resample_records_to_1hz(
                records,
                duration_s=70,
                pad_to_duration=pad_to_full_window,
            )
        else:
            records = resample_records_to_1hz(
                records,
                duration_s=FULL_WINDOW_SIZE,
                pad_to_duration=pad_to_full_window,
            )
    elif mode == "fast":
        if normalize_window:
            records = normalize_window_records(records, 70)
    else:
        if pad_to_full_window:
            records = pad_session_to_full_window(records, FULL_WINDOW_SIZE)
        elif normalize_window:
            records = normalize_window_records(records, FULL_WINDOW_SIZE)

    if mode == "fast":
        X = features_for_fast_window(records)
    else:
        X = features_for_latest_window(records, FULL_WINDOW_SIZE)

    if X is None:
        return None

    model = bundle["model"]
    expected_columns = bundle.get("feature_columns")
    if expected_columns is not None:
        X = X.reindex(columns=expected_columns, fill_value=0.0)

    labels = bundle.get("labels")
    if labels is None:
        labels = list(model.classes_)

    probs = model.predict_proba(X)[0]
    best_idx = int(probs.argmax())

    return {
        "label": labels[best_idx],
        "confidence": float(probs[best_idx]),
        "probs": dict(zip(labels, [float(p) for p in probs])),
    }


def compute_baseline(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Baseline for the readiness monitor.

    This intentionally uses only the first READY_BASELINE_END_S seconds,
    not the whole 0-25 s used by the fast ML model, because users may start
    moving the sample close to the nozzle near the end of the baseline phase.
    """
    if not records:
        return {}

    df = pd.DataFrame(records).copy()
    if "elapsed_s" not in df.columns:
        return {}

    baseline_df = df[
        (df["elapsed_s"] >= 0)
        & (df["elapsed_s"] < READY_BASELINE_END_S)
    ].copy()

    # Require at least a few readings so we do not lock a poor baseline.
    if baseline_df.empty or len(baseline_df) < 5:
        return {}

    baseline: dict[str, float] = {}
    for col in SENSOR_COLUMNS:
        if col in baseline_df.columns:
            value = pd.to_numeric(baseline_df[col], errors="coerce").mean()
            if pd.notna(value):
                baseline[col] = float(value)

    return baseline


def latest_window(records: list[dict[str, Any]], seconds: float) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).copy()
    if "elapsed_s" not in df.columns:
        return df.tail(int(seconds))

    max_t = float(df["elapsed_s"].max())
    return df[df["elapsed_s"] >= max_t - seconds].copy()


def is_ready_for_next_sample(
    monitor_records: list[dict[str, Any]],
    baseline: dict[str, float],
) -> tuple[bool, dict[str, Any]]:
    """
    Monitorização contínua, sem usar classes/sessões de recovery.

    Compara os últimos ~30 s com a baseline inicial da medição.
    """
    if not baseline:
        return False, {
            "state": "Sem baseline",
            "reason": "Ainda não há baseline suficiente para comparar.",
            "checks": {},
            "details": {},
        }

    window = latest_window(monitor_records, RECOVERY_STABLE_SECONDS)

    if window.empty or len(window) < max(5, int(RECOVERY_STABLE_SECONDS / 2)):
        return False, {
            "state": "A aguardar",
            "reason": f"Ainda não há {RECOVERY_STABLE_SECONDS:.0f} s de dados recentes.",
            "checks": {},
            "details": {},
        }

    recent: dict[str, float] = {}
    for col in SENSOR_COLUMNS:
        if col in window.columns:
            value = pd.to_numeric(window[col], errors="coerce").mean()
            if pd.notna(value):
                recent[col] = float(value)

    checks: dict[str, bool] = {}
    details: dict[str, float] = {}

    def add_abs_check(sensor: str, name: str, threshold_key: str) -> None:
        if sensor not in baseline or sensor not in recent:
            return
        delta = abs(recent[sensor] - baseline[sensor])
        details[threshold_key] = delta
        checks[name] = delta <= READY_THRESHOLDS[threshold_key]

    add_abs_check("voc", "VOC perto da baseline", "voc_abs_delta")
    add_abs_check("mq135_voltage", "MQ135 perto da baseline", "mq135_abs_delta")
    add_abs_check("mq3_voltage", "MQ3 perto da baseline", "mq3_abs_delta")
    add_abs_check("bme688_iaq", "BME688 IAQ perto da baseline", "bme688_iaq_abs_delta")
    add_abs_check("bme688_breath_voc_equivalent", "BME688 bVOC perto da baseline", "bme688_bvoc_abs_delta")
    add_abs_check("bme688_co2_equivalent", "BME688 eCO2 perto da baseline", "bme688_eco2_abs_delta")

    if (
        "bme688_gas_resistance" in baseline
        and "bme688_gas_resistance" in recent
        and abs(baseline["bme688_gas_resistance"]) > 1e-9
    ):
        gas_rel_delta = abs(
            recent["bme688_gas_resistance"] - baseline["bme688_gas_resistance"]
        ) / abs(baseline["bme688_gas_resistance"])
        details["bme688_gas_relative_delta"] = gas_rel_delta
        checks["BME688 gas resistance estabilizada"] = (
            gas_rel_delta <= READY_THRESHOLDS["bme688_gas_relative_delta"]
        )

    if not checks:
        return False, {
            "state": "Sem sensores suficientes",
            "reason": "Não há sensores suficientes para avaliar recuperação.",
            "checks": checks,
            "details": details,
        }

    ready = all(checks.values())
    if ready:
        return True, {
            "state": "Pronta / clean air",
            "reason": f"Sensores estáveis durante ~{RECOVERY_STABLE_SECONDS:.0f} s.",
            "checks": checks,
            "details": details,
        }

    failed = [name for name, ok in checks.items() if not ok]
    return False, {
        "state": "A recuperar",
        "reason": "Ainda há sinais afastados da baseline: " + ", ".join(failed),
        "checks": checks,
        "details": details,
    }


def air_state_from_continuous_classifier(
    *,
    continuous_prediction: dict[str, Any] | None,
    history: list[dict[str, Any]],
    running: bool,
    elapsed_s: float,
) -> tuple[bool, dict[str, Any]]:
    """
    Determine whether the air looks clean enough for the next sample.

    This is intentionally based on the continuous classifier, not on strict
    return-to-baseline thresholds. It is more practical for sensors that have
    long recovery tails, such as BME688 bVOC after alcohol/lemon.
    """
    if continuous_prediction is None:
        if running:
            return False, {
                "state": "A medir",
                "short_state": "A medir",
                "reason": "A classificação contínua ainda não tem dados suficientes.",
                "clean_air_probability": None,
            }

        return False, {
            "state": "A aguardar",
            "short_state": "A aguardar",
            "reason": "A aguardar dados suficientes para avaliar se o ar voltou a clean_air.",
            "clean_air_probability": None,
        }

    probs = continuous_prediction.get("probs", {})
    clean_air_prob = float(probs.get("clean_air", 0.0))
    current_label = str(continuous_prediction.get("label", "—"))
    current_confidence = float(continuous_prediction.get("confidence", 0.0))

    # During the controlled sample session, do not claim it is ready even if
    # clean_air temporarily appears.
    if running and elapsed_s < SESSION_END_S:
        return False, {
            "state": "Medição em curso",
            "short_state": "A medir",
            "reason": "A sessão de amostra ainda está em curso.",
            "clean_air_probability": clean_air_prob,
            "current_label": current_label,
            "current_confidence": current_confidence,
        }

    stable_history = []
    if history:
        latest_t = float(history[-1].get("elapsed_s", 0.0))
        stable_history = [
            row for row in history
            if latest_t - float(row.get("elapsed_s", 0.0)) <= CLEAN_AIR_STABLE_SECONDS
        ]

    stable_clean_air = False
    if stable_history:
        stable_clean_air = all(
            float(row.get("clean_air", 0.0)) >= CLEAN_AIR_READY_CONFIDENCE
            for row in stable_history
        )

    if stable_clean_air and clean_air_prob >= CLEAN_AIR_READY_CONFIDENCE:
        return True, {
            "state": "Ar limpo",
            "short_state": "Ar limpo",
            "reason": (
                f"O classificador contínuo reconhece clean_air com confiança alta "
                f"há ~{CLEAN_AIR_STABLE_SECONDS:.0f} s."
            ),
            "clean_air_probability": clean_air_prob,
            "current_label": current_label,
            "current_confidence": current_confidence,
        }

    if clean_air_prob >= CLEAN_AIR_READY_CONFIDENCE:
        return False, {
            "state": "Quase limpo",
            "short_state": "Quase limpo",
            "reason": (
                "O ar já parece clean_air, mas ainda precisa de se manter estável "
                f"durante ~{CLEAN_AIR_STABLE_SECONDS:.0f} s."
            ),
            "clean_air_probability": clean_air_prob,
            "current_label": current_label,
            "current_confidence": current_confidence,
        }

    if clean_air_prob >= CLEAN_AIR_ALMOST_CONFIDENCE or current_label == "clean_air":
        return False, {
            "state": "A estabilizar",
            "short_state": "A estabilizar",
            "reason": "O ar está a aproximar-se de clean_air, mas a confiança ainda é moderada.",
            "clean_air_probability": clean_air_prob,
            "current_label": current_label,
            "current_confidence": current_confidence,
        }

    return False, {
        "state": "A recuperar",
        "short_state": "A recuperar",
        "reason": (
            f"O classificador contínuo ainda vê '{current_label}' "
            f"com {current_confidence * 100:.1f}% de confiança."
        ),
        "clean_air_probability": clean_air_prob,
        "current_label": current_label,
        "current_confidence": current_confidence,
    }


def reset_measurement() -> None:
    now = time.time()

    # Reset only the sample session.
    # Continuous classification remains active and keeps its own history.
    st.session_state.session_version = int(st.session_state.get("session_version", 0)) + 1
    st.session_state.session_records = []
    st.session_state.final_preview_history = []

    st.session_state.measurement_start_time = now
    st.session_state.running = True
    st.session_state.session_finished = False

    st.session_state.final_prediction_locked = None
    st.session_state.fast_prediction_locked = None
    st.session_state.baseline_locked = {}


def stop_measurement() -> None:
    # Stop only the current sample session.
    # Continuous classification remains active.
    st.session_state.running = False


def status_text(elapsed_s: float, finished: bool) -> str:
    if finished:
        return "Sessão concluída. Previsão final congelada; monitorização contínua ativa."
    if elapsed_s < READY_BASELINE_END_S:
        return "Baseline para recuperação: manter ar limpo junto ao nozzle."
    if elapsed_s < 35:
        return "Transição: aproximar a amostra do nozzle."
    if elapsed_s < 60:
        return "Exposição inicial: previsão rápida quase pronta."
    if elapsed_s < 90:
        return "Exposição: manter a amostra junto ao nozzle."
    if elapsed_s < 180:
        return "Recuperação: remover a amostra e deixar a câmara recuperar."
    return "Sessão completa: a previsão final será congelada."




def render_probability_bars(probs: dict[str, float]) -> None:
    """
    Stable probability chart for the dashboard.

    Uses Altair instead of many st.progress widgets, avoiding duplicated visual
    elements after fast reruns/session resets. Falls back to a dataframe if
    Altair is unavailable.
    """
    if not probs:
        st.info("Sem probabilidades para mostrar.")
        return

    df = pd.DataFrame(
        [
            {"classe": str(label), "probabilidade": float(prob) * 100.0}
            for label, prob in probs.items()
        ]
    ).sort_values("probabilidade", ascending=False)

    if alt is None:
        st.dataframe(
            df.assign(probabilidade=df["probabilidade"].map(lambda v: f"{v:.1f}%")),
            use_container_width=True,
            hide_index=True,
            height=190,
        )
        return

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4)
        .encode(
            y=alt.Y("classe:N", sort="-x", title=None),
            x=alt.X(
                "probabilidade:Q",
                title=None,
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(format=".0f", labelExpr="datum.label + '%'"),
            ),
            tooltip=[
                alt.Tooltip("classe:N", title="Classe"),
                alt.Tooltip("probabilidade:Q", title="Confiança", format=".1f"),
            ],
        )
        .properties(height=max(130, 27 * len(df)))
    )

    text = (
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", dx=4, fontSize=12)
        .encode(
            y=alt.Y("classe:N", sort="-x", title=None),
            x=alt.X("probabilidade:Q", scale=alt.Scale(domain=[0, 100])),
            text=alt.Text("probabilidade:Q", format=".1f"),
        )
    )

    st.altair_chart(chart + text, use_container_width=True)


def labels_from_bundle(bundle: dict[str, Any] | None) -> list[str]:
    if bundle is None:
        return ["alcohol", "clean_air", "coffee", "vinegar"]
    labels = bundle.get("labels")
    if labels is not None:
        return list(labels)
    model = bundle.get("model")
    if model is not None and hasattr(model, "classes_"):
        return list(model.classes_)
    return ["alcohol", "clean_air", "coffee", "vinegar"]


def empty_probs_for_bundle(bundle: dict[str, Any] | None) -> dict[str, float]:
    return {label: 0.0 for label in labels_from_bundle(bundle)}


def probs_or_empty(
    prediction: dict[str, Any] | None,
    bundle: dict[str, Any] | None,
) -> dict[str, float]:
    if prediction is not None:
        return prediction.get("probs", {})
    return empty_probs_for_bundle(bundle)


def render_waiting_metric(title: str, message: str) -> None:
    st.metric(title, "A aguardar", message)



def sensor_value(record: dict[str, Any] | None, key: str) -> float | None:
    if not record:
        return None
    value = record.get(key)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def delta_from_baseline(
    latest_record: dict[str, Any] | None,
    baseline: dict[str, float],
    key: str,
) -> float | None:
    current = sensor_value(latest_record, key)
    base = baseline.get(key)
    if current is None or base is None:
        return None
    return current - float(base)


def build_system_messages(
    *,
    latest_record: dict[str, Any] | None,
    baseline: dict[str, float],
    elapsed_s: float,
    running: bool,
    session_finished: bool,
    ready: bool,
    fast_prediction: dict[str, Any] | None,
    final_prediction: dict[str, Any] | None,
    continuous_prediction: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """
    Human-readable messages for participants.

    These messages are intentionally rule-based and explanatory. They do not
    use recovery classes. They summarize what the sensors/model appear to be
    doing in real time.
    """
    messages: list[dict[str, str]] = []

    if latest_record is None:
        return [
            {
                "level": "info",
                "title": "A aguardar leituras",
                "body": "Ainda não há dados suficientes dos sensores.",
            }
        ]

    if running:
        if elapsed_s < READY_BASELINE_END_S:
            messages.append(
                {
                    "level": "info",
                    "title": "A medir o ar de referência",
                    "body": "Mantém ar limpo junto ao nozzle para criar uma baseline estável.",
                }
            )
        elif elapsed_s < 35:
            messages.append(
                {
                    "level": "info",
                    "title": "Aproximação da amostra",
                    "body": "A amostra pode ser aproximada do nozzle. O sistema começa a procurar alterações nos sensores.",
                }
            )
        elif elapsed_s < 90:
            messages.append(
                {
                    "level": "info",
                    "title": "Exposição em curso",
                    "body": "Os sensores estão a registar a assinatura inicial da amostra.",
                }
            )
        elif elapsed_s < SESSION_END_S:
            messages.append(
                {
                    "level": "info",
                    "title": "Fase de recuperação",
                    "body": "Remove a amostra e deixa a câmara recuperar. A previsão final será congelada aos 180 s.",
                }
            )
    elif session_finished:
        messages.append(
            {
                "level": "success" if ready else "warning",
                "title": "Sessão concluída",
                "body": (
                    "A câmara parece pronta para uma nova amostra."
                    if ready
                    else "A previsão final está congelada, mas a câmara ainda está a recuperar."
                ),
            }
        )
    else:
        messages.append(
            {
                "level": "info",
                "title": "Monitorização contínua ativa",
                "body": "O sistema está a observar o ar em tempo real. Inicia uma nova medição quando quiseres testar uma amostra.",
            }
        )

    # Sensor-based observations compared with baseline.
    voc_delta = delta_from_baseline(latest_record, baseline, "voc")
    mq135_delta = delta_from_baseline(latest_record, baseline, "mq135_voltage")
    mq3_delta = delta_from_baseline(latest_record, baseline, "mq3_voltage")
    iaq_delta = delta_from_baseline(latest_record, baseline, "bme688_iaq")
    bvoc_delta = delta_from_baseline(latest_record, baseline, "bme688_breath_voc_equivalent")

    gas_current = sensor_value(latest_record, "bme688_gas_resistance")
    gas_base = baseline.get("bme688_gas_resistance") if baseline else None
    gas_rel_drop = None
    if gas_current is not None and gas_base is not None and abs(gas_base) > 1e-9:
        gas_rel_drop = (float(gas_base) - gas_current) / abs(float(gas_base))

    signal_parts: list[str] = []

    if voc_delta is not None and voc_delta > 25:
        signal_parts.append(f"VOC subiu +{voc_delta:.0f}")
    elif voc_delta is not None and voc_delta > 10:
        signal_parts.append(f"VOC subiu ligeiramente (+{voc_delta:.0f})")

    if mq135_delta is not None and mq135_delta > 0.05:
        signal_parts.append(f"MQ135 subiu +{mq135_delta:.3f} V")
    if mq3_delta is not None and mq3_delta > 0.03:
        signal_parts.append(f"MQ3 subiu +{mq3_delta:.3f} V")

    if iaq_delta is not None and iaq_delta > 40:
        signal_parts.append(f"IAQ aumentou +{iaq_delta:.0f}")
    elif iaq_delta is not None and iaq_delta > 15:
        signal_parts.append(f"IAQ aumentou +{iaq_delta:.0f}")

    if bvoc_delta is not None and bvoc_delta > 10:
        signal_parts.append(f"bVOC subiu +{bvoc_delta:.1f} ppm")

    if gas_rel_drop is not None and gas_rel_drop > 0.25:
        signal_parts.append(f"resistência de gás caiu {gas_rel_drop * 100:.0f}%")
    elif gas_rel_drop is not None and gas_rel_drop > 0.12:
        signal_parts.append(f"resistência de gás caiu {gas_rel_drop * 100:.0f}%")

    if signal_parts:
        messages.append(
            {
                "level": "warning",
                "title": "Alteração química detetada",
                "body": "Os sensores afastaram-se da baseline: " + "; ".join(signal_parts[:4]) + ".",
            }
        )
    elif baseline:
        messages.append(
            {
                "level": "success",
                "title": "Sinais próximos da baseline",
                "body": "Neste momento os principais sensores parecem próximos do ar de referência.",
            }
        )

    # Model observations.
    active_prediction = final_prediction or continuous_prediction or fast_prediction
    if active_prediction is not None:
        label = active_prediction.get("label", "—")
        confidence = float(active_prediction.get("confidence", 0.0)) * 100

        if confidence >= 85:
            messages.append(
                {
                    "level": "success",
                    "title": "Modelo confiante",
                    "body": f"O modelo está bastante confiante em '{label}' ({confidence:.1f}%).",
                }
            )
        elif confidence >= 60:
            messages.append(
                {
                    "level": "info",
                    "title": "Modelo em observação",
                    "body": f"A classe mais provável é '{label}', mas a confiança ainda é moderada ({confidence:.1f}%).",
                }
            )
        else:
            messages.append(
                {
                    "level": "warning",
                    "title": "Previsão ainda ambígua",
                    "body": f"O modelo ainda não está muito confiante. Classe atual: '{label}' ({confidence:.1f}%).",
                }
            )

    # Special cases useful for the project.
    if bvoc_delta is not None and bvoc_delta > 200:
        messages.append(
            {
                "level": "warning",
                "title": "BME688 bVOC muito elevado",
                "body": "O bVOC está muito alto. Algumas amostras, como limão ou álcool, podem saturar este indicador e demorar a recuperar.",
            }
        )

    # Keep the box compact.
    return messages[:5]


def render_system_messages(messages: list[dict[str, str]]) -> None:
    for message in messages:
        level = message.get("level", "info")
        title = message.get("title", "")
        body = message.get("body", "")

        text = f"**{title}**  \n{body}"

        if level == "success":
            st.success(text)
        elif level == "warning":
            st.warning(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)



def render_live_session_timer(
    *,
    start_time: float,
    running: bool,
    finished: bool,
    current_elapsed: float,
    session_end_s: float = SESSION_END_S,
) -> None:
    """
    Browser-side timer.

    Streamlit reruns can take 1-5 seconds when charts/model predictions are
    being redrawn. This component keeps the visible timer moving smoothly in
    the browser between backend reruns, so the app does not look frozen.
    """
    start_ms = int(start_time * 1000)
    frozen_elapsed = min(float(current_elapsed), float(session_end_s))
    frozen_elapsed_ms = int(frozen_elapsed * 1000)

    running_js = "true" if running and not finished else "false"

    components.html(
        f"""
        <div style="
            background:#f8fafc;
            border:1px solid #e5e7eb;
            border-radius:0.75rem;
            padding:0.55rem 0.75rem;
            height:72px;
            box-sizing:border-box;
            font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        ">
            <div style="
                font-size:0.78rem;
                color:#475569;
                margin-bottom:0.15rem;
                font-weight:600;
            ">
                Tempo
            </div>
            <div id="live-session-timer" style="
                font-size:1.65rem;
                line-height:1.1;
                font-weight:700;
                color:#0f172a;
            ">
                {int(frozen_elapsed)} s
            </div>
            <div id="live-session-subtitle" style="
                font-size:0.72rem;
                color:#64748b;
                margin-top:0.1rem;
            ">
                temporizador local
            </div>
        </div>

        <script>
        const startMs = {start_ms};
        const running = {running_js};
        const frozenElapsedMs = {frozen_elapsed_ms};
        const sessionEndMs = {int(session_end_s * 1000)};

        function updateTimer() {{
            let elapsedMs;
            if (running) {{
                elapsedMs = Date.now() - startMs;
            }} else {{
                elapsedMs = frozenElapsedMs;
            }}

            elapsedMs = Math.max(0, Math.min(elapsedMs, sessionEndMs));

            const elapsedS = Math.floor(elapsedMs / 1000);
            const timer = document.getElementById("live-session-timer");
            const subtitle = document.getElementById("live-session-subtitle");

            if (timer) {{
                timer.textContent = elapsedS + " s";
            }}

            if (subtitle) {{
                if (running) {{
                    subtitle.textContent = "a atualizar no ecrã";
                }} else {{
                    subtitle.textContent = "sessão parada/concluída";
                }}
            }}
        }}

        updateTimer();
        setInterval(updateTimer, 250);
        </script>
        """,
        height=78,
    )


def render_live_session_progress(
    *,
    start_time: float,
    running: bool,
    finished: bool,
    current_elapsed: float,
    session_end_s: float = SESSION_END_S,
) -> None:
    """
    Browser-side progress bar synchronized with the live timer.

    This avoids the progress bar only moving when Streamlit reruns.
    """
    start_ms = int(start_time * 1000)
    frozen_elapsed = min(float(current_elapsed), float(session_end_s))
    frozen_elapsed_ms = int(frozen_elapsed * 1000)
    running_js = "true" if running and not finished else "false"

    components.html(
        f"""
        <div style="
            width:100%;
            height:14px;
            background:#e5e7eb;
            border-radius:999px;
            overflow:hidden;
            margin:2px 0 6px 0;
            border:1px solid #d1d5db;
            box-sizing:border-box;
            font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        ">
            <div id="live-session-progress-fill" style="
                height:100%;
                width:0%;
                background:linear-gradient(90deg,#10b981,#3b82f6);
                border-radius:999px;
                transition:width 0.2s linear;
            "></div>
        </div>

        <script>
        const progressStartMs = {start_ms};
        const progressRunning = {running_js};
        const progressFrozenElapsedMs = {frozen_elapsed_ms};
        const progressSessionEndMs = {int(session_end_s * 1000)};

        function updateProgress() {{
            let elapsedMs;
            if (progressRunning) {{
                elapsedMs = Date.now() - progressStartMs;
            }} else {{
                elapsedMs = progressFrozenElapsedMs;
            }}

            elapsedMs = Math.max(0, Math.min(elapsedMs, progressSessionEndMs));
            const pct = (elapsedMs / progressSessionEndMs) * 100.0;

            const fill = document.getElementById("live-session-progress-fill");
            if (fill) {{
                fill.style.width = pct.toFixed(2) + "%";
            }}
        }}

        updateProgress();
        setInterval(updateProgress, 250);
        </script>
        """,
        height=24,
    )



def sample_instruction(elapsed_s: float, finished: bool) -> dict[str, str]:
    """
    Large but compact instruction banner for the sample collection protocol.
    """
    if finished:
        return {
            "icon": "✅",
            "phase": "SESSÃO CONCLUÍDA",
            "instruction": "Resultado final congelado. Aguarda até o estado do ar voltar a limpo.",
            "color": "#ecfdf5",
            "border": "#10b981",
        }

    if elapsed_s < 20:
        return {
            "icon": "🟢",
            "phase": "0–20 s · MEDIR AR LIMPO",
            "instruction": "Mantém o nozzle sem amostra para criar a referência inicial.",
            "color": "#ecfdf5",
            "border": "#10b981",
        }

    if elapsed_s < 35:
        return {
            "icon": "🟡",
            "phase": "20–35 s · APROXIMAR AMOSTRA",
            "instruction": "Aproxima a amostra do nozzle, mantendo a mesma distância.",
            "color": "#fffbeb",
            "border": "#f59e0b",
        }

    if elapsed_s < 90:
        return {
            "icon": "🔴",
            "phase": "35–90 s · MANTER AMOSTRA",
            "instruction": "Mantém a amostra junto ao nozzle. O sistema está a observar a assinatura do cheiro.",
            "color": "#fef2f2",
            "border": "#ef4444",
        }

    if elapsed_s < 180:
        return {
            "icon": "🔵",
            "phase": "90–180 s · REMOVER AMOSTRA",
            "instruction": "Remove a amostra e deixa a câmara recuperar.",
            "color": "#eff6ff",
            "border": "#3b82f6",
        }

    return {
        "icon": "✅",
        "phase": "SESSÃO CONCLUÍDA",
        "instruction": "Resultado final congelado. A classificação contínua continua ativa.",
        "color": "#ecfdf5",
        "border": "#10b981",
    }


def render_sample_instruction(elapsed_s: float, finished: bool) -> None:
    msg = sample_instruction(elapsed_s, finished)

    st.markdown(
        f"""
        <div style="
            background: {msg['color']};
            border-left: 8px solid {msg['border']};
            border-radius: 12px;
            padding: 10px 16px;
            margin: 4px 0 8px 0;
            display: flex;
            align-items: center;
            gap: 14px;
        ">
            <div style="font-size: 2.0rem; line-height: 1;">
                {msg['icon']}
            </div>
            <div>
                <div style="
                    font-size: 1.15rem;
                    font-weight: 800;
                    line-height: 1.1;
                ">
                    {msg['phase']}
                </div>
                <div style="
                    font-size: 1.0rem;
                    line-height: 1.2;
                    margin-top: 2px;
                ">
                    {msg['instruction']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

defaults = {
    "session_records": [],
    "monitor_records": [],
    "continuous_prediction_history": [],
    "final_preview_history": [],
    "measurement_start_time": time.time(),
    "monitor_start_time": time.time(),
    "running": False,
    "session_finished": False,
    "final_prediction_locked": None,
    "fast_prediction_locked": None,
    "baseline_locked": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Demo defaults. Configuration is intentionally not shown at the top,
# to keep the dashboard clean for the TV/demo view.
# The browser-side timer stays smooth; 750 ms avoids overloading Streamlit reruns.
read_interval_ms = 750
broker = "10.0.0.2"
port = 1883
username = ""
password = ""
full_model_path = DEFAULT_FULL_MODEL_PATH
fast_model_path = DEFAULT_FAST_MODEL_PATH


# -------------------------------------------------------------------
# Load models and MQTT
# -------------------------------------------------------------------

full_bundle = load_model_bundle(full_model_path)
fast_bundle = load_model_bundle(fast_model_path)

reader = create_mqtt_reader(
    broker=broker,
    port=int(port),
    username=username or None,
    password=password or None,
    topics_tuple=tuple(sorted(DEFAULT_MQTT_TOPICS.items())),
)


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------

header_left, header_right = st.columns([0.55, 5.45])

with header_left:
    logo_path = Path(LOGO_PATH)
    if logo_path.exists():
        st.image(str(logo_path), width=115)
    else:
        st.markdown("## 👃")

with header_right:
    st.markdown("# Electronic Nose")
    st.markdown(
        "<span class='small-caption'>IA + sensores ambientais para reconhecer padrões de odor.</span>",
        unsafe_allow_html=True,
    )

st.divider()

control_a, control_b, control_c = st.columns([1.2, 1.2, 3.2])

with control_a:
    if st.button("▶️ Iniciar nova medição", type="primary", use_container_width=True):
        reset_measurement()
        st.rerun()
        st.stop()

with control_b:
    if st.button("⏹️ Terminar sessão", use_container_width=True):
        stop_measurement()
        st.rerun()
        st.stop()

with control_c:
    st.info(
        "Protocolo: 0–20 s ar limpo, 20–35 s aproximar amostra, "
        "35–90 s exposição, 90–180 s recuperação. Aos 180 s a previsão final congela."
    )


# -------------------------------------------------------------------
# Read one sample from MQTT
# -------------------------------------------------------------------

# Continuous monitoring is always active while the app is open.
latest = reader.read()
now = time.time()

monitor_elapsed_s = now - st.session_state.monitor_start_time
session_elapsed_s = now - st.session_state.measurement_start_time

monitor_row = {
    "timestamp": pd.Timestamp.utcnow().isoformat(),
    "elapsed_s": monitor_elapsed_s,
}
for col in SENSOR_COLUMNS:
    monitor_row[col] = latest.get(col)

st.session_state.monitor_records.append(monitor_row)
st.session_state.monitor_records = st.session_state.monitor_records[-1800:]

# Session records are only collected during an explicit sample session.
if st.session_state.running and session_elapsed_s <= SESSION_END_S:
    session_row = dict(monitor_row)
    session_row["elapsed_s"] = session_elapsed_s
    st.session_state.session_records.append(session_row)
    st.session_state.session_records = st.session_state.session_records[-600:]


session_records = st.session_state.session_records
monitor_records = st.session_state.monitor_records

# elapsed = current sample-session time.
# monitor_records use an independent continuous clock.
elapsed = 0.0
if st.session_state.running or st.session_state.session_finished:
    elapsed = time.time() - st.session_state.measurement_start_time
elif session_records:
    elapsed = float(session_records[-1].get("elapsed_s", 0.0))

session_elapsed = 0.0
if session_records:
    session_elapsed = float(session_records[-1].get("elapsed_s", 0.0))


# -------------------------------------------------------------------
# Predictions and automatic final lock
# -------------------------------------------------------------------

fast_prediction = predict_from_records(session_records, fast_bundle, mode="fast")

# Preview using the final/full model before 180 s.
# This is useful pedagogically, but it is NOT the official final result.
final_preview_prediction = None
if st.session_state.running and len(session_records) >= 60:
    final_preview_prediction = predict_from_records(
        session_records,
        full_bundle,
        mode="full",
        pad_to_full_window=True,
    )

    if final_preview_prediction is not None:
        preview_row = {
            "elapsed_s": float(session_records[-1].get("elapsed_s", 0.0)),
            "label": final_preview_prediction["label"],
            "confidence": final_preview_prediction["confidence"],
        }
        for label, prob in final_preview_prediction["probs"].items():
            preview_row[label] = prob

        if (
            not st.session_state.final_preview_history
            or preview_row["elapsed_s"] > st.session_state.final_preview_history[-1]["elapsed_s"]
        ):
            st.session_state.final_preview_history.append(preview_row)
            st.session_state.final_preview_history = st.session_state.final_preview_history[-240:]

if fast_prediction is not None:
    st.session_state.fast_prediction_locked = fast_prediction

if (
    not st.session_state.baseline_locked
    and session_records
    and float(session_records[-1].get("elapsed_s", 0.0)) >= READY_BASELINE_END_S
):
    baseline = compute_baseline(session_records)
    if baseline:
        st.session_state.baseline_locked = baseline

if st.session_state.running and elapsed >= SESSION_END_S:
    # At 180 s, Streamlit often has the last stored real sample at ~179.x s.
    # Use the real elapsed time to trigger the lock, then pad/resample the
    # recorded session into a complete 0..179 s window before predicting.
    full_prediction_for_lock = predict_from_records(
        session_records,
        full_bundle,
        mode="full",
        pad_to_full_window=True,
    )

    if full_prediction_for_lock is not None:
        st.session_state.final_prediction_locked = full_prediction_for_lock

    if not st.session_state.baseline_locked:
        st.session_state.baseline_locked = compute_baseline(session_records)

    st.session_state.running = False
    st.session_state.session_finished = True

display_fast_prediction = (
    st.session_state.fast_prediction_locked
    if st.session_state.fast_prediction_locked is not None
    else fast_prediction
)

display_full_prediction = st.session_state.final_prediction_locked


# -------------------------------------------------------------------
# Continuous diagnostic classification during recovery
# -------------------------------------------------------------------

continuous_prediction = None
continuous_mode = None

# Real-time panel logic:
# - during an active sample session, keep the right-side panel aligned with the
#   same session data used by the fast/final preview models;
# - outside a session, use the true rolling continuous classifier to monitor
#   recovery / clean_air.
if st.session_state.running:
    if final_preview_prediction is not None:
        continuous_prediction = final_preview_prediction
        continuous_mode = "session_final_preview"
    elif fast_prediction is not None:
        continuous_prediction = fast_prediction
        continuous_mode = "session_fast_preview"
elif monitor_records and float(monitor_records[-1].get("elapsed_s", 0.0)) >= FULL_WINDOW_SIZE:
    continuous_prediction = predict_from_records(
        rolling_records_last_seconds(monitor_records, FULL_WINDOW_SIZE),
        full_bundle,
        mode="full",
        regularize_1hz=True,
    )
    continuous_mode = "rolling_full_180s"
elif monitor_records and float(monitor_records[-1].get("elapsed_s", 0.0)) >= 60:
    continuous_prediction = predict_from_records(
        rolling_records_last_seconds(monitor_records, 70),
        fast_bundle,
        mode="fast",
        regularize_1hz=True,
    )
    continuous_mode = "rolling_fast_60s"

if continuous_prediction is not None and continuous_mode in {"rolling_full_180s", "rolling_fast_60s"}:
    history_row = {
        "elapsed_s": float(monitor_records[-1].get("elapsed_s", 0.0)),
        "label": continuous_prediction["label"],
        "confidence": continuous_prediction["confidence"],
        "mode": continuous_mode,
    }
    for label, prob in continuous_prediction["probs"].items():
        history_row[label] = prob

    # Avoid appending duplicate points caused by reruns without a new timestamp.
    if (
        not st.session_state.continuous_prediction_history
        or history_row["elapsed_s"] > st.session_state.continuous_prediction_history[-1]["elapsed_s"]
    ):
        st.session_state.continuous_prediction_history.append(history_row)
        st.session_state.continuous_prediction_history = st.session_state.continuous_prediction_history[-900:]


# -------------------------------------------------------------------
# Dashboard UI
# -------------------------------------------------------------------

baseline_ready, baseline_ready_info = is_ready_for_next_sample(
    monitor_records=monitor_records,
    baseline=st.session_state.baseline_locked,
)

ready, ready_info = air_state_from_continuous_classifier(
    continuous_prediction=continuous_prediction,
    history=st.session_state.continuous_prediction_history,
    running=st.session_state.running,
    elapsed_s=elapsed,
)

effective_points = len(
    resample_records_to_1hz(
        session_records,
        duration_s=FULL_WINDOW_SIZE,
        pad_to_duration=False,
    )
)

metric_cols = st.columns(5)

with metric_cols[0]:
    render_live_session_timer(
        start_time=st.session_state.measurement_start_time,
        running=st.session_state.running,
        finished=st.session_state.session_finished,
        current_elapsed=elapsed,
    )

with metric_cols[1]:
    st.metric("Leituras", f"{len(session_records)} / {effective_points}")

with metric_cols[2]:
    st.metric("Rápida", "Pronta" if display_fast_prediction is not None else "A aguardar")

with metric_cols[3]:
    st.metric(
        "Final",
        "Congelada" if display_full_prediction is not None else "A aguardar",
    )

with metric_cols[4]:
    st.metric("Estado do ar", ready_info.get("short_state", ready_info.get("state", "—")))

render_live_session_progress(
    start_time=st.session_state.measurement_start_time,
    running=st.session_state.running,
    finished=st.session_state.session_finished,
    current_elapsed=elapsed,
)
render_sample_instruction(elapsed, st.session_state.session_finished)

if display_full_prediction is not None:
    st.success("Sessão concluída: a previsão final foi congelada aos 180 s.")

dashboard_tab, graphs_tab, details_tab = st.tabs(
    ["Dashboard", "Gráficos", "Detalhes"]
)

with dashboard_tab:
    dash_left, dash_mid, dash_right, dash_messages = st.columns([1.15, 1.15, 1.15, 1.0])

    with dash_left:
        st.markdown("### ⚡ Previsão rápida")
        if fast_bundle is None:
            st.warning(f"Modelo rápido não encontrado: `{fast_model_path}`")
            render_probability_bars(empty_probs_for_bundle(None))
        elif display_fast_prediction is None:
            render_waiting_metric("Resultado preliminar", "~60 s")
            render_probability_bars(empty_probs_for_bundle(fast_bundle))
        else:
            st.metric(
                "Resultado preliminar",
                display_fast_prediction["label"],
                f"{display_fast_prediction['confidence'] * 100:.1f}% confiança",
            )
            render_probability_bars(display_fast_prediction["probs"])

    with dash_mid:
        st.markdown("### 🎯 Modelo final")
        if full_bundle is None:
            st.warning(f"Modelo completo não encontrado: `{full_model_path}`")
            render_probability_bars(empty_probs_for_bundle(None))
        elif display_full_prediction is not None:
            st.metric(
                "Final congelado",
                display_full_prediction["label"],
                f"{display_full_prediction['confidence'] * 100:.1f}% confiança",
            )
            st.caption("Resultado oficial congelado aos 180 s.")
            render_probability_bars(display_full_prediction["probs"])
        elif final_preview_prediction is not None:
            st.metric(
                "Pré-visualização",
                final_preview_prediction["label"],
                f"{final_preview_prediction['confidence'] * 100:.1f}% confiança",
            )
            st.caption("Pode mudar até aos 180 s.")
            render_probability_bars(final_preview_prediction["probs"])
        else:
            render_waiting_metric("Pré-visualização", "~60 s")
            st.caption("O resultado oficial congela aos 180 s.")
            render_probability_bars(empty_probs_for_bundle(full_bundle))

    with dash_right:
        panel_title = "🧭 Estado da sessão" if st.session_state.running else "🔄 Classificação contínua"
        st.markdown(f"### {panel_title}")

        if full_bundle is None and fast_bundle is None:
            st.warning("Modelos não encontrados.")
            render_probability_bars(empty_probs_for_bundle(None))
        elif continuous_prediction is None:
            if st.session_state.running:
                missing = max(0, int(60 - elapsed))
                render_waiting_metric("Sessão atual", f"~{missing} s")
                st.caption("Durante a sessão, este painel usa os mesmos dados da amostra atual.")
            else:
                current_elapsed = float(monitor_records[-1].get("elapsed_s", 0.0)) if monitor_records else 0.0
                missing = max(0, int(60 - current_elapsed))
                render_waiting_metric("Tempo real", f"~{missing} s")
                st.caption("Fora da sessão, usa uma janela deslizante contínua.")
            render_probability_bars(empty_probs_for_bundle(fast_bundle or full_bundle))
        else:
            metric_label = "Sessão atual" if st.session_state.running else "Tempo real"
            st.metric(
                metric_label,
                continuous_prediction["label"],
                f"{continuous_prediction['confidence'] * 100:.1f}% confiança",
            )

            if continuous_mode == "session_final_preview":
                st.caption("Alinhado com a pré-visualização do modelo final.")
            elif continuous_mode == "session_fast_preview":
                st.caption("Alinhado com a previsão rápida da sessão.")
            elif continuous_mode == "rolling_full_180s":
                st.caption("Janela deslizante completa de ~180 s.")
            else:
                st.caption("Janela rápida inicial de ~60 s.")

            render_probability_bars(continuous_prediction["probs"])

    with dash_messages:
        st.markdown("### 💬 Mensagens")

        latest_record_for_messages = monitor_records[-1] if monitor_records else None
        system_messages = build_system_messages(
            latest_record=latest_record_for_messages,
            baseline=st.session_state.baseline_locked,
            elapsed_s=elapsed,
            running=st.session_state.running,
            session_finished=st.session_state.session_finished,
            ready=ready,
            fast_prediction=display_fast_prediction,
            final_prediction=display_full_prediction,
            continuous_prediction=continuous_prediction,
        )

        # Compact dashboard: show the most relevant messages in the side column.
        render_system_messages(system_messages[:4])



with graphs_tab:
    graph_left, graph_right = st.columns([1.25, 1.0])

    with graph_left:
        st.markdown("### 📈 Sensores da sessão")
        if len(session_records) >= 2:
            df_session = pd.DataFrame(session_records)
            chart_cols = [
                "voc",
                "mq135_voltage",
                "mq3_voltage",
                "bme688_gas_resistance",
                "bme688_iaq",
                "bme688_breath_voc_equivalent",
                "scd40_co2",
            ]
            available_chart_cols = [c for c in chart_cols if c in df_session.columns]
            selected_cols = st.multiselect(
                "Sensores",
                options=available_chart_cols,
                default=available_chart_cols[:3],
            )
            if selected_cols:
                st.line_chart(df_session.set_index("elapsed_s")[selected_cols], height=260)
        else:
            st.info("O gráfico aparece depois de pelo menos duas leituras.")

    with graph_right:
        st.markdown("### 🔄 Probabilidades contínuas")
        history = st.session_state.continuous_prediction_history
        if len(history) >= 2:
            history_df = pd.DataFrame(history)
            prob_cols = [
                col for col in history_df.columns
                if col not in {"elapsed_s", "label", "confidence", "mode"}
            ]
            if prob_cols:
                st.line_chart(history_df.set_index("elapsed_s")[prob_cols], height=260)
        else:
            st.info("A evolução aparece após algumas previsões contínuas.")

    with st.expander("Histórico contínuo / recuperação", expanded=False):
        if len(monitor_records) >= 2:
            df_monitor = pd.DataFrame(monitor_records)
            recovery_cols = [
                "voc",
                "mq135_voltage",
                "mq3_voltage",
                "bme688_iaq",
                "bme688_breath_voc_equivalent",
                "bme688_gas_resistance",
            ]
            available_recovery_cols = [c for c in recovery_cols if c in df_monitor.columns]
            if available_recovery_cols:
                st.line_chart(df_monitor.set_index("elapsed_s")[available_recovery_cols], height=230)
        else:
            st.write("Sem dados contínuos suficientes.")

with details_tab:
    details_left, details_right = st.columns(2)

    with details_left:
        st.markdown("### Leituras atuais")
        if monitor_records:
            latest_record = monitor_records[-1]
            sensor_df = pd.DataFrame(
                [{"sensor": col, "value": latest_record.get(col)} for col in SENSOR_COLUMNS]
            )
            st.dataframe(sensor_df, use_container_width=True, hide_index=True, height=330)
        else:
            st.info("Ainda não há leituras.")

    with details_right:
        st.markdown("### Estado do ar")

        if ready:
            st.success(ready_info.get("state", "Ar limpo"))
            st.metric("Pronto para nova amostra?", "Sim")
        else:
            st.warning(ready_info.get("state", "A recuperar"))
            st.metric("Pronto para nova amostra?", "Aguarde")

        st.caption(ready_info.get("reason", ""))

        clean_air_prob = ready_info.get("clean_air_probability")
        if clean_air_prob is not None:
            st.write(f"Probabilidade de clean_air: **{clean_air_prob * 100:.1f}%**")


    with st.expander("⚙️ Configuração avançada", expanded=False):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            st.write("**Leitura**")
            st.write(f"Intervalo de leitura/backend: `{read_interval_ms} ms`")
            st.write(f"MQTT broker: `{broker}`")
            st.write(f"MQTT port: `{port}`")

        with cfg_col2:
            st.write("**Modelos**")
            st.write(f"Modelo completo: `{full_model_path}`")
            st.write(f"Modelo rápido: `{fast_model_path}`")

        with cfg_col3:
            st.write("**Monitorização**")
            st.write("Não usa recovery sessions/classes.")
            st.write("Reamostragem interna para 1 Hz.")
            with st.expander("Limites de recuperação", expanded=False):
                st.json(READY_THRESHOLDS)

        with st.expander("Tópicos MQTT", expanded=False):
            for key, topic in DEFAULT_MQTT_TOPICS.items():
                st.code(f"{key}: {topic}", language="text")

    with st.expander("Dados da sessão e downloads", expanded=False):
        if session_records:
            df_session_records = pd.DataFrame(session_records)
            st.write("Últimas leituras da sessão")
            st.dataframe(df_session_records.tail(20), use_container_width=True)

            st.download_button(
                "Descarregar sessão em CSV",
                data=df_session_records.to_csv(index=False).encode("utf-8"),
                file_name="electronic_nose_session.csv",
                mime="text/csv",
            )
        else:
            st.write("Sem dados da sessão.")

        if monitor_records:
            df_monitor_records = pd.DataFrame(monitor_records)
            st.write("Últimas leituras contínuas")
            st.dataframe(df_monitor_records.tail(20), use_container_width=True)

        if st.session_state.final_preview_history:
            st.write("Histórico da pré-visualização do modelo final")
            st.dataframe(pd.DataFrame(st.session_state.final_preview_history).tail(20), use_container_width=True)

        if st.session_state.continuous_prediction_history:
            st.write("Histórico de classificação contínua")
            st.dataframe(pd.DataFrame(st.session_state.continuous_prediction_history).tail(20), use_container_width=True)


# -------------------------------------------------------------------
# Auto refresh
# -------------------------------------------------------------------

# Always refresh so continuous classification remains active even outside a session.
time.sleep(read_interval_ms / 1000.0)
st.rerun()
