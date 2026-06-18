from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import tomllib
except Exception:
    tomllib = None

import joblib
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import SENSOR_COLUMNS, DEFAULT_MQTT_TOPICS
    from src.features import features_for_fast_window
    from src.mqtt_listener import MQTTConfig, MQTTLiveReader
except ImportError:
    from config import SENSOR_COLUMNS, DEFAULT_MQTT_TOPICS
    from features import features_for_fast_window
    from mqtt_listener import MQTTConfig, MQTTLiveReader


# ---------------------------------------------------------------------
# Basic page
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Electronic Nose",
    page_icon="👃",
    layout="wide",
)

st.markdown(
    """
    <style>
        #MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"],
        div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], .stDeployButton {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        .block-container {
            padding-top: 0.55rem !important;
            padding-bottom: 1rem !important;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 0.75rem;
            padding: 0.55rem 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "app_config.toml"
DEFAULT_MODEL_PATH = "models/electronic_nose_fast_lemon.joblib"
DEFAULT_LOGO_PATH = "assets/electronic_nose_logo_app_light.webp"
DEFAULT_VISTALAB_LOGO_PATH = "assets/vistalab_logo_padded.png"

DEFAULTS: dict[str, Any] = {
    "app": {
        "read_interval_ms": 1000,
        "logo_path": DEFAULT_LOGO_PATH,
        "vistalab_logo_path": DEFAULT_VISTALAB_LOGO_PATH,
        "show_logos": True,
    },
    "mqtt": {
        "broker": "10.0.0.2",
        "port": 1883,
        "username": "",
        "password": "",
    },
    "models": {
        "model_path": DEFAULT_MODEL_PATH,
    },
    "session": {
        "preview_start_s": 60,
        "lock_s": 90,
        "end_s": 180,
        "clean_air_stable_s": 25,
        "clean_air_ready_probability": 0.75,
        "clean_air_almost_probability": 0.55,
    },
    "ui": {
        "max_messages": 4,
        "show_graphs_by_default": False,
    },
    "topics": dict(DEFAULT_MQTT_TOPICS),
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path_value: str | Path) -> tuple[dict[str, Any], Path | None, str | None]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return DEFAULTS, None, None

    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".toml":
            if tomllib is None:
                return DEFAULTS, None, "TOML não está disponível nesta versão de Python."
            loaded = tomllib.loads(raw)
        elif path.suffix.lower() == ".json":
            loaded = json.loads(raw)
        else:
            return DEFAULTS, None, "Formato de configuração não suportado. Usa TOML ou JSON."

        if not isinstance(loaded, dict):
            return DEFAULTS, None, "O ficheiro de configuração deve conter um dicionário."

        return deep_update(DEFAULTS, loaded), path, None
    except Exception as exc:
        return DEFAULTS, None, f"Erro ao ler configuração: {exc}"


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


config_path = os.environ.get("APP_CONFIG", DEFAULT_CONFIG_PATH)
config, loaded_config_path, config_warning = load_config(config_path)

app_cfg = config["app"]
mqtt_cfg = config["mqtt"]
model_cfg = config["models"]
session_cfg = config["session"]
ui_cfg = config["ui"]

READ_INTERVAL_MS = int(app_cfg.get("read_interval_ms", 1000))
LOGO_PATH = str(app_cfg.get("logo_path", DEFAULT_LOGO_PATH))
VISTALAB_LOGO_PATH = str(app_cfg.get("vistalab_logo_path", DEFAULT_VISTALAB_LOGO_PATH))
SHOW_LOGOS = bool(app_cfg.get("show_logos", True))

MODEL_PATH = str(model_cfg.get("model_path", DEFAULT_MODEL_PATH))
TOPICS = dict(DEFAULT_MQTT_TOPICS)
TOPICS.update(config.get("topics", {}) or {})

PREVIEW_START_S = int(session_cfg.get("preview_start_s", 60))
LOCK_S = int(session_cfg.get("lock_s", 90))
END_S = int(session_cfg.get("end_s", 180))
CLEAN_AIR_STABLE_S = int(session_cfg.get("clean_air_stable_s", 25))
CLEAN_AIR_READY_PROB = float(session_cfg.get("clean_air_ready_probability", 0.75))
CLEAN_AIR_ALMOST_PROB = float(session_cfg.get("clean_air_almost_probability", 0.55))
MAX_MESSAGES = int(ui_cfg.get("max_messages", 4))


# ---------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------

@st.cache_resource
def load_model_bundle(path_value: str) -> dict[str, Any] | None:
    path = resolve_path(path_value)
    if not path.exists():
        return None

    loaded = joblib.load(path)

    if isinstance(loaded, dict) and "model" in loaded:
        bundle = loaded
    else:
        bundle = {"model": loaded}

    model = bundle["model"]
    if "labels" not in bundle and hasattr(model, "classes_"):
        bundle["labels"] = list(model.classes_)

    return bundle


@st.cache_resource
def create_mqtt_reader(
    broker: str,
    port: int,
    username: str,
    password: str,
    topics_tuple: tuple[tuple[str, str], ...],
) -> MQTTLiveReader:
    kwargs: dict[str, Any] = {
        "broker": broker,
        "port": int(port),
        "topics": dict(topics_tuple),
    }
    if username:
        kwargs["username"] = username
    if password:
        kwargs["password"] = password

    reader = MQTTLiveReader(MQTTConfig(**kwargs))
    reader.start()
    return reader


model_bundle = load_model_bundle(MODEL_PATH)
reader = create_mqtt_reader(
    broker=str(mqtt_cfg.get("broker", "10.0.0.2")),
    port=int(mqtt_cfg.get("port", 1883)),
    username=str(mqtt_cfg.get("username", "") or ""),
    password=str(mqtt_cfg.get("password", "") or ""),
    topics_tuple=tuple(sorted(TOPICS.items())),
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "running": False,
        "finished": False,
        "session_rows": [],
        "monitor_rows": [],
        "preview_history": [],
        "continuous_history": [],
        "final_prediction": None,
        "session_id": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_session() -> None:
    st.session_state.running = True
    st.session_state.finished = False
    st.session_state.session_rows = []
    st.session_state.preview_history = []
    st.session_state.final_prediction = None
    st.session_state.session_id = int(st.session_state.get("session_id", 0)) + 1


def stop_session() -> None:
    st.session_state.running = False
    st.session_state.finished = True


init_state()


# ---------------------------------------------------------------------
# Data / model utilities
# ---------------------------------------------------------------------

def numeric_row_from_latest(latest: dict[str, Any], elapsed_s: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_s": int(elapsed_s),
    }
    for col in SENSOR_COLUMNS:
        value = latest.get(col)
        try:
            row[col] = float(value) if value is not None else None
        except Exception:
            row[col] = None
    return row


def resample_to_1hz(
    rows: list[dict[str, Any]],
    *,
    duration_s: int,
    pad_to_duration: bool,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    df = pd.DataFrame(rows).copy()
    if "elapsed_s" not in df.columns:
        df["elapsed_s"] = range(len(df))

    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df = df.dropna(subset=["elapsed_s"]).copy()
    # pandas.merge_asof requires both merge keys to have exactly the same dtype.
    # Session elapsed_s can arrive as int while the target timeline is float.
    df["elapsed_s"] = df["elapsed_s"].astype(float)
    df = df.sort_values("elapsed_s")
    if df.empty:
        return []

    end_s = duration_s - 1 if pad_to_duration else min(duration_s - 1, int(df["elapsed_s"].max()))
    if end_s < 1:
        return []

    target = pd.DataFrame({"elapsed_s": [float(i) for i in range(end_s + 1)]})
    target["elapsed_s"] = target["elapsed_s"].astype(float)
    merged = pd.merge_asof(target, df, on="elapsed_s", direction="backward")
    merged = merged.ffill().bfill()
    merged["elapsed_s"] = target["elapsed_s"]
    return merged.to_dict(orient="records")


def normalize_rolling_window(rows: list[dict[str, Any]], seconds: int) -> list[dict[str, Any]]:
    if not rows:
        return []

    selected = rows[-seconds:]
    normalized = []
    for idx, row in enumerate(selected):
        new_row = dict(row)
        new_row["elapsed_s"] = float(idx)
        normalized.append(new_row)
    return normalized


def model_labels(bundle: dict[str, Any] | None) -> list[str]:
    if bundle is None:
        return []
    if bundle.get("labels"):
        return [str(x) for x in bundle["labels"]]
    model = bundle.get("model")
    if model is not None and hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]
    return []


def empty_probs(bundle: dict[str, Any] | None) -> dict[str, float]:
    return {label: 0.0 for label in model_labels(bundle)}


def predict_fast(
    rows: list[dict[str, Any]],
    bundle: dict[str, Any] | None,
    *,
    duration_s: int,
    pad_to_duration: bool,
) -> dict[str, Any] | None:
    if bundle is None or not rows:
        return None

    prepared = resample_to_1hz(rows, duration_s=duration_s, pad_to_duration=pad_to_duration)
    if not prepared:
        return None

    try:
        X = features_for_fast_window(prepared)
    except Exception as exc:
        return {
            "label": "erro",
            "confidence": 0.0,
            "probs": {},
            "error": f"Erro ao extrair features: {exc}",
        }

    if X is None:
        return None

    model = bundle["model"]
    expected_columns = bundle.get("feature_columns")
    if expected_columns is not None:
        X = X.reindex(columns=expected_columns, fill_value=0.0)

    labels = model_labels(bundle)

    try:
        if hasattr(model, "predict_proba"):
            probs_raw = model.predict_proba(X)[0]
            best_idx = int(probs_raw.argmax())
            probs = dict(zip(labels, [float(p) for p in probs_raw]))
            return {
                "label": labels[best_idx],
                "confidence": float(probs_raw[best_idx]),
                "probs": probs,
            }

        label = str(model.predict(X)[0])
        return {"label": label, "confidence": 1.0, "probs": {label: 1.0}}
    except Exception as exc:
        return {
            "label": "erro",
            "confidence": 0.0,
            "probs": {},
            "error": f"Erro na previsão: {exc}",
        }


def append_prediction_history(history_key: str, elapsed_s: int, prediction: dict[str, Any]) -> None:
    if prediction is None or prediction.get("label") == "erro":
        return

    row = {
        "elapsed_s": int(elapsed_s),
        "label": prediction["label"],
        "confidence": float(prediction["confidence"]),
    }
    for label, prob in prediction.get("probs", {}).items():
        row[label] = float(prob)

    history = st.session_state[history_key]
    if not history or int(history[-1].get("elapsed_s", -1)) < int(elapsed_s):
        history.append(row)
        st.session_state[history_key] = history[-600:]


def recovery_state(
    continuous_prediction: dict[str, Any] | None,
    history: list[dict[str, Any]],
    running: bool,
    sample_elapsed_s: int,
) -> dict[str, Any]:
    if running and sample_elapsed_s < LOCK_S:
        return {
            "state": "Medição em curso",
            "short": "A medir",
            "clean_air_probability": None,
            "dominant": "—",
            "confidence": None,
            "reason": "A amostra ainda está em avaliação.",
        }

    if continuous_prediction is None:
        return {
            "state": "Sem dados suficientes",
            "short": "A aguardar",
            "clean_air_probability": None,
            "dominant": "—",
            "confidence": None,
            "reason": "A recuperação aparece quando houver dados contínuos suficientes.",
        }

    probs = continuous_prediction.get("probs", {})
    clean_air_prob = float(probs.get("clean_air", 0.0))
    dominant = str(continuous_prediction.get("label", "—"))
    confidence = float(continuous_prediction.get("confidence", 0.0))

    stable_rows = history[-CLEAN_AIR_STABLE_S:]
    stable_clean = (
        len(stable_rows) >= max(5, CLEAN_AIR_STABLE_S // 2)
        and all(float(row.get("clean_air", 0.0)) >= CLEAN_AIR_READY_PROB for row in stable_rows)
    )

    if stable_clean and clean_air_prob >= CLEAN_AIR_READY_PROB:
        state = "Ar limpo"
        short = "Ar limpo"
        reason = "Clean_air está estável com confiança alta."
    elif clean_air_prob >= CLEAN_AIR_READY_PROB:
        state = "Quase limpo"
        short = "Quase limpo"
        reason = "Clean_air está alto, mas ainda precisa de estabilidade."
    elif clean_air_prob >= CLEAN_AIR_ALMOST_PROB or dominant == "clean_air":
        state = "A estabilizar"
        short = "A estabilizar"
        reason = "O ar está a aproximar-se de clean_air."
    else:
        state = "A recuperar"
        short = "A recuperar"
        reason = f"O classificador ainda vê {dominant}."

    return {
        "state": state,
        "short": short,
        "clean_air_probability": clean_air_prob,
        "dominant": dominant,
        "confidence": confidence,
        "reason": reason,
    }


def build_messages(
    *,
    latest: dict[str, Any],
    sample_elapsed_s: int,
    running: bool,
    finished: bool,
    final_prediction: dict[str, Any] | None,
    recovery: dict[str, Any],
    model_ok: bool,
) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []

    if not model_ok:
        messages.append(("error", f"Modelo não encontrado: {MODEL_PATH}"))

    if latest:
        missing = [col for col in SENSOR_COLUMNS if latest.get(col) is None]
        if len(missing) == len(SENSOR_COLUMNS):
            messages.append(("warning", "Sem leituras MQTT válidas neste momento."))
    else:
        messages.append(("warning", "Ainda não chegou nenhuma leitura MQTT."))

    if running:
        if sample_elapsed_s < 20:
            messages.append(("info", "0–20 s: manter ar limpo junto ao nozzle."))
        elif sample_elapsed_s < 35:
            messages.append(("warning", "20–35 s: aproximar a amostra de forma estável."))
        elif sample_elapsed_s < PREVIEW_START_S:
            messages.append(("warning", "35–60 s: manter a amostra junto ao nozzle."))
        elif sample_elapsed_s < LOCK_S:
            messages.append(("info", "60–90 s: previsão em avaliação, ainda não é final."))
        elif sample_elapsed_s < END_S:
            messages.append(("success", "Classificação pronta. Remover a amostra e deixar recuperar."))
    elif finished:
        messages.append(("success", "Sessão concluída. A classificação ficou congelada."))
    else:
        messages.append(("info", "Pronto para iniciar nova medição."))

    if final_prediction is not None:
        messages.append((
            "success",
            f"Resultado final: {final_prediction['label']} ({final_prediction['confidence'] * 100:.1f}%).",
        ))

    messages.append(("info", f"Recuperação: {recovery['state']} — {recovery['reason']}"))

    return messages[:MAX_MESSAGES]


# ---------------------------------------------------------------------
# Read one MQTT sample per refresh
# ---------------------------------------------------------------------

latest = reader.read() or {}

monitor_elapsed_s = len(st.session_state.monitor_rows)
monitor_row = numeric_row_from_latest(latest, monitor_elapsed_s)
st.session_state.monitor_rows.append(monitor_row)
st.session_state.monitor_rows = st.session_state.monitor_rows[-900:]

if st.session_state.running:
    sample_elapsed_s = len(st.session_state.session_rows)
    if sample_elapsed_s <= END_S:
        session_row = numeric_row_from_latest(latest, sample_elapsed_s)
        st.session_state.session_rows.append(session_row)
        st.session_state.session_rows = st.session_state.session_rows[-(END_S + 30):]
    if len(st.session_state.session_rows) - 1 >= END_S:
        st.session_state.running = False
        st.session_state.finished = True

session_rows = st.session_state.session_rows
monitor_rows = st.session_state.monitor_rows
sample_elapsed_s = max(0, len(session_rows) - 1)


# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------

preview_prediction = None
final_prediction = st.session_state.final_prediction

if st.session_state.running and sample_elapsed_s >= PREVIEW_START_S and sample_elapsed_s < LOCK_S:
    preview_prediction = predict_fast(
        session_rows,
        model_bundle,
        duration_s=LOCK_S,
        pad_to_duration=True,
    )
    if preview_prediction and preview_prediction.get("label") != "erro":
        append_prediction_history("preview_history", sample_elapsed_s, preview_prediction)

if sample_elapsed_s >= LOCK_S and st.session_state.final_prediction is None and session_rows:
    final_prediction = predict_fast(
        session_rows,
        model_bundle,
        duration_s=LOCK_S,
        pad_to_duration=True,
    )
    if final_prediction and final_prediction.get("label") != "erro":
        st.session_state.final_prediction = final_prediction
    else:
        final_prediction = None

final_prediction = st.session_state.final_prediction
sample_prediction = final_prediction or preview_prediction

continuous_prediction = None
if len(monitor_rows) >= PREVIEW_START_S:
    rolling_rows = normalize_rolling_window(monitor_rows, LOCK_S)
    continuous_prediction = predict_fast(
        rolling_rows,
        model_bundle,
        duration_s=LOCK_S,
        pad_to_duration=True,
    )
    if continuous_prediction and continuous_prediction.get("label") != "erro":
        append_prediction_history("continuous_history", len(monitor_rows), continuous_prediction)

recovery = recovery_state(
    continuous_prediction,
    st.session_state.continuous_history,
    st.session_state.running,
    sample_elapsed_s,
)


# ---------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------

def render_probability_list(probs: dict[str, float]) -> None:
    if not probs:
        st.info("Sem probabilidades disponíveis.")
        return

    for label, value in sorted(probs.items(), key=lambda item: float(item[1]), reverse=True):
        p = max(0.0, min(1.0, float(value)))
        st.progress(p, text=f"{label} — {p * 100:.1f}%")


def render_messages(messages: list[tuple[str, str]]) -> None:
    for level, text in messages:
        if level == "success":
            st.success(text)
        elif level == "warning":
            st.warning(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

if config_warning:
    st.warning(config_warning)

header_left, header_right = st.columns([3.5, 1.2], vertical_alignment="center")

with header_left:
    logo_col, title_col = st.columns([0.18, 0.82], vertical_alignment="center")
    with logo_col:
        logo_path = resolve_path(LOGO_PATH)
        if SHOW_LOGOS and logo_path.exists():
            st.image(str(logo_path), width=96)
        else:
            st.markdown("## 👃")
    with title_col:
        st.markdown("## Electronic Nose")
        st.caption("IA + sensores ambientais para reconhecer padrões de odor.")

with header_right:
    vistalab_logo_path = resolve_path(VISTALAB_LOGO_PATH)
    if SHOW_LOGOS and vistalab_logo_path.exists():
        st.image(str(vistalab_logo_path), width=190)
    elif SHOW_LOGOS:
        st.caption("VISTALab · Centro ALGORITMI")

st.divider()

controls_a, controls_b, controls_c = st.columns([1, 1, 3], vertical_alignment="center")
with controls_a:
    st.button("▶️ Iniciar medição", type="primary", width="stretch", on_click=start_session)
with controls_b:
    st.button("⏹️ Terminar", width="stretch", on_click=stop_session)
with controls_c:
    st.caption(
        "Protocolo: 0–20 s ar limpo · 20–35 s aproximar · 35–90 s manter amostra · "
        "90–180 s recuperação. Usa apenas o modelo rápido."
    )

status_cols = st.columns(5)
with status_cols[0]:
    st.metric("Tempo", f"{sample_elapsed_s:03d} s")
with status_cols[1]:
    st.metric("Leituras", f"{len(session_rows)}")
with status_cols[2]:
    if final_prediction:
        classification_state = "Pronta"
    elif preview_prediction:
        classification_state = "Em avaliação"
    elif st.session_state.running:
        classification_state = "A recolher"
    else:
        classification_state = "A aguardar"
    st.metric("Classificação", classification_state)
with status_cols[3]:
    session_state = "Em curso" if st.session_state.running else ("Concluída" if st.session_state.finished else "Parada")
    st.metric("Sessão", session_state)
with status_cols[4]:
    st.metric("Câmara", recovery["short"])

progress_value = min(sample_elapsed_s / max(END_S, 1), 1.0)
st.progress(progress_value, text=f"Progresso da sessão: {sample_elapsed_s}/{END_S} s")

messages = build_messages(
    latest=latest,
    sample_elapsed_s=sample_elapsed_s,
    running=st.session_state.running,
    finished=st.session_state.finished,
    final_prediction=final_prediction,
    recovery=recovery,
    model_ok=model_bundle is not None,
)

left, middle, right = st.columns([1, 1, 1])

with left:
    st.markdown("### 🧠 Classificação da amostra")
    if model_bundle is None:
        st.error(f"Modelo rápido não encontrado: `{MODEL_PATH}`")
        render_probability_list(empty_probs(model_bundle))
    elif sample_prediction is None:
        if st.session_state.running and sample_elapsed_s < PREVIEW_START_S:
            remaining = PREVIEW_START_S - sample_elapsed_s
            st.metric("Resultado", "A aguardar", f"~{remaining} s")
        elif st.session_state.running:
            st.metric("Resultado", "A recolher dados")
        else:
            st.metric("Resultado", "—")
        render_probability_list(empty_probs(model_bundle))
    else:
        label = sample_prediction.get("label", "—")
        confidence = float(sample_prediction.get("confidence", 0.0))
        if sample_prediction.get("error"):
            st.error(sample_prediction["error"])
        else:
            title = "Resultado final" if final_prediction else "Resultado em avaliação"
            st.metric(title, label, f"{confidence * 100:.1f}% confiança")
            if not final_prediction:
                st.caption(f"Atualiza até congelar aos {LOCK_S} s.")
        render_probability_list(sample_prediction.get("probs", {}))

with middle:
    st.markdown("### 🌬️ Recuperação da câmara")
    clean_air_prob = recovery.get("clean_air_probability")
    if clean_air_prob is None:
        st.metric("Estado", recovery["state"])
        st.caption(recovery["reason"])
    else:
        st.metric("Estado", recovery["state"], f"clean_air {clean_air_prob * 100:.1f}%")
        st.progress(clean_air_prob, text=f"Probabilidade de clean_air — {clean_air_prob * 100:.1f}%")
        st.write(f"Classe contínua dominante: **{recovery['dominant']}**")
        if recovery.get("confidence") is not None:
            st.write(f"Confiança: **{float(recovery['confidence']) * 100:.1f}%**")
        st.caption(recovery["reason"])

with right:
    st.markdown("### 💬 Mensagens")
    render_messages(messages)

show_graphs = st.checkbox(
    "Mostrar gráficos",
    value=bool(ui_cfg.get("show_graphs_by_default", False)),
)

if show_graphs:
    graph_a, graph_b = st.columns(2)

    with graph_a:
        st.markdown("#### Sensores da sessão")
        if len(session_rows) >= 2:
            df_session = pd.DataFrame(session_rows)
            sensor_cols = [
                "voc",
                "mq135_voltage",
                "mq3_voltage",
                "bme688_iaq",
                "bme688_breath_voc_equivalent",
                "bme688_gas_resistance",
                "scd40_co2",
            ]
            available = [c for c in sensor_cols if c in df_session.columns]
            selected = st.multiselect("Sensores", available, default=available[:3])
            if selected:
                st.line_chart(df_session.set_index("elapsed_s")[selected], height=260)
        else:
            st.info("O gráfico aparece depois de pelo menos duas leituras.")

    with graph_b:
        st.markdown("#### Probabilidades contínuas")
        if len(st.session_state.continuous_history) >= 2:
            df_hist = pd.DataFrame(st.session_state.continuous_history)
            prob_cols = [
                col for col in df_hist.columns
                if col not in {"elapsed_s", "label", "confidence"}
            ]
            if prob_cols:
                st.line_chart(df_hist.set_index("elapsed_s")[prob_cols], height=260)
        else:
            st.info("A evolução aparece após algumas previsões contínuas.")

    with st.expander("Histórico contínuo / recuperação por escalas"):
        if len(monitor_rows) >= 2:
            df_monitor = pd.DataFrame(monitor_rows)
            groups = [
                ("VOC / SGP41", ["voc"]),
                ("MQ voltages", ["mq135_voltage", "mq3_voltage"]),
                ("BME688 IAQ / bVOC / eCO₂", [
                    "bme688_iaq",
                    "bme688_breath_voc_equivalent",
                    "bme688_co2_equivalent",
                ]),
                ("BME688 gas resistance", ["bme688_gas_resistance"]),
                ("SCD40 CO₂", ["scd40_co2"]),
            ]

            for idx in range(0, len(groups), 2):
                cols = st.columns(2)
                for col_ui, (title, names) in zip(cols, groups[idx:idx + 2]):
                    available = [name for name in names if name in df_monitor.columns]
                    if available:
                        with col_ui:
                            st.markdown(f"**{title}**")
                            st.line_chart(df_monitor.set_index("elapsed_s")[available], height=190)
        else:
            st.info("Sem dados contínuos suficientes.")

with st.expander("Detalhes técnicos"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Configuração**")
        st.write(f"Ficheiro: `{loaded_config_path or 'valores por defeito'}`")
        st.write(f"Refresh: `{READ_INTERVAL_MS} ms`")
        st.write(f"Modelo: `{MODEL_PATH}`")
    with col2:
        st.write("**Sessão**")
        st.write(f"Pré-visualização: `{PREVIEW_START_S} s`")
        st.write(f"Bloqueio: `{LOCK_S} s`")
        st.write(f"Fim: `{END_S} s`")
    with col3:
        st.write("**MQTT**")
        st.write(f"Broker: `{mqtt_cfg.get('broker')}`")
        st.write(f"Porta: `{mqtt_cfg.get('port')}`")


# ---------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------

if st_autorefresh is not None:
    st_autorefresh(
        interval=READ_INTERVAL_MS,
        key=f"fresh_app_refresh_{st.session_state.get('session_id', 0)}",
    )
else:
    st.caption("Para atualização automática instala: pip install streamlit-autorefresh")
    st.button("Atualizar agora")
