from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

try:
    from src.config import SENSOR_COLUMNS, DEFAULT_MQTT_TOPICS
    from src.features import features_for_latest_window, features_for_fast_window
    from src.mqtt_listener import MQTTConfig, MQTTLiveReader
except ImportError:
    from config import SENSOR_COLUMNS, DEFAULT_MQTT_TOPICS
    from features import features_for_latest_window, features_for_fast_window
    from mqtt_listener import MQTTConfig, MQTTLiveReader


# -----------------------------
# App settings
# -----------------------------

st.set_page_config(
    page_title="Electronic Nose",
    page_icon="👃",
    layout="wide",
)

DEFAULT_FULL_MODEL_PATH = "models/electronic_nose_indoor_outdoor.joblib"
DEFAULT_FAST_MODEL_PATH = "models/electronic_nose_fast.joblib"

FULL_WINDOW_SIZE = 180
FAST_WINDOW_SIZE = 70


# -----------------------------
# Helpers
# -----------------------------

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
    topics = dict(topics_tuple)

    cfg_kwargs = {
        "broker": broker,
        "port": port,
        "topics": topics,
    }

    if username:
        cfg_kwargs["username"] = username
    if password:
        cfg_kwargs["password"] = password

    cfg = MQTTConfig(**cfg_kwargs)
    reader = MQTTLiveReader(cfg)
    reader.start()
    return reader


def predict_from_records(
    records: list[dict[str, Any]],
    bundle: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any] | None:
    if bundle is None:
        return None

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


def reset_measurement() -> None:
    st.session_state.records = []
    st.session_state.measurement_start_time = time.time()
    st.session_state.running = True


def stop_measurement() -> None:
    st.session_state.running = False


def status_text(elapsed_s: float) -> str:
    if elapsed_s < 25:
        return "Baseline: manter ar limpo junto ao nozzle."
    if elapsed_s < 35:
        return "Transição: aproximar a amostra do nozzle."
    if elapsed_s < 60:
        return "Exposição inicial: previsão rápida quase pronta."
    if elapsed_s < 90:
        return "Exposição: manter a amostra junto ao nozzle."
    if elapsed_s < 180:
        return "Recuperação: remover a amostra e deixar a câmara recuperar."
    return "Sessão completa: previsão final disponível."


def confidence_badge(prediction: dict[str, Any] | None) -> str:
    if prediction is None:
        return "—"
    return f"{prediction['label']} ({prediction['confidence'] * 100:.1f}%)"


# -----------------------------
# Session state
# -----------------------------

if "records" not in st.session_state:
    st.session_state.records = []

if "measurement_start_time" not in st.session_state:
    st.session_state.measurement_start_time = time.time()

if "running" not in st.session_state:
    st.session_state.running = False


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Configuração")

read_interval_ms = st.sidebar.number_input(
    "Intervalo de leitura (ms)",
    min_value=250,
    max_value=10000,
    value=1000,
    step=250,
    help="Tempo entre atualizações da app. 1000 ms = 1 segundo.",
)

broker = st.sidebar.text_input("MQTT broker", value="10.0.0.2")
port = st.sidebar.number_input("MQTT port", min_value=1, max_value=65535, value=1883)
username = st.sidebar.text_input("MQTT username", value="")
password = st.sidebar.text_input("MQTT password", value="", type="password")

full_model_path = st.sidebar.text_input(
    "Modelo completo",
    value=DEFAULT_FULL_MODEL_PATH,
)

fast_model_path = st.sidebar.text_input(
    "Modelo rápido",
    value=DEFAULT_FAST_MODEL_PATH,
)

st.sidebar.caption("Tópicos MQTT")
with st.sidebar.expander("Ver tópicos"):
    for key, topic in DEFAULT_MQTT_TOPICS.items():
        st.code(f"{key}: {topic}", language="text")


# -----------------------------
# Load models and MQTT
# -----------------------------

full_bundle = load_model_bundle(full_model_path)
fast_bundle = load_model_bundle(fast_model_path)

topics_tuple = tuple(sorted(DEFAULT_MQTT_TOPICS.items()))

reader = create_mqtt_reader(
    broker=broker,
    port=int(port),
    username=username or None,
    password=password or None,
    topics_tuple=topics_tuple,
)


# -----------------------------
# Main UI
# -----------------------------

st.title("👃 Electronic Nose")
st.caption("IA + sensores ambientais para reconhecer padrões de odor.")

col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    if st.button("Iniciar nova medição", type="primary", use_container_width=True):
        reset_measurement()

with col_b:
    if st.button("Parar", use_container_width=True):
        stop_measurement()

with col_c:
    st.info(
        "Protocolo recomendado: 0–25 s ar limpo, 25–35 s aproximar amostra, "
        "35–90 s exposição, 90–180 s recuperação."
    )


# -----------------------------
# Read one sample from MQTT
# -----------------------------

if st.session_state.running:
    latest = reader.read()

    elapsed_s = time.time() - st.session_state.measurement_start_time

    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "elapsed_s": elapsed_s,
    }

    for col in SENSOR_COLUMNS:
        row[col] = latest.get(col)

    st.session_state.records.append(row)

    # Keep enough history but avoid unbounded growth
    st.session_state.records = st.session_state.records[-600:]


records = st.session_state.records
elapsed = 0.0
if records:
    elapsed = float(records[-1].get("elapsed_s", 0.0))


# -----------------------------
# Status
# -----------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Tempo da sessão", f"{elapsed:.0f} s")

with m2:
    st.metric("Leituras", len(records))

with m3:
    st.metric("Previsão rápida", "Pronta" if len(records) >= 60 else "A aguardar")

with m4:
    st.metric("Previsão final", "Pronta" if len(records) >= FULL_WINDOW_SIZE else "A aguardar")

st.progress(min(elapsed / 180.0, 1.0))
st.write(f"**Estado:** {status_text(elapsed)}")


# -----------------------------
# Predictions
# -----------------------------

fast_prediction = predict_from_records(records, fast_bundle, mode="fast")
full_prediction = predict_from_records(records, full_bundle, mode="full")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.subheader("Previsão rápida (~60 s)")

    if fast_bundle is None:
        st.warning(f"Modelo rápido não encontrado: `{fast_model_path}`")
    elif fast_prediction is None:
        st.info("A aguardar dados suficientes para previsão rápida.")
    else:
        st.metric(
            "Resultado preliminar",
            fast_prediction["label"],
            f"{fast_prediction['confidence'] * 100:.1f}% confiança",
        )
        st.bar_chart(pd.Series(fast_prediction["probs"]).sort_values(ascending=False))

with pred_col2:
    st.subheader("Previsão final (~180 s)")

    if full_bundle is None:
        st.warning(f"Modelo completo não encontrado: `{full_model_path}`")
    elif full_prediction is None:
        st.info("A aguardar dados suficientes para previsão final.")
    else:
        st.metric(
            "Resultado final",
            full_prediction["label"],
            f"{full_prediction['confidence'] * 100:.1f}% confiança",
        )
        st.bar_chart(pd.Series(full_prediction["probs"]).sort_values(ascending=False))


# -----------------------------
# Latest sensor readings
# -----------------------------

st.subheader("Leituras atuais dos sensores")

if records:
    latest_record = records[-1]
    sensor_values = {
        col: latest_record.get(col)
        for col in SENSOR_COLUMNS
    }

    sensor_df = pd.DataFrame(
        [{"sensor": key, "value": value} for key, value in sensor_values.items()]
    )

    st.dataframe(sensor_df, use_container_width=True, hide_index=True)
else:
    st.info("Ainda não há leituras nesta medição.")


# -----------------------------
# Charts
# -----------------------------

st.subheader("Histórico da sessão")

if len(records) >= 2:
    df = pd.DataFrame(records)

    chart_cols = [
        "voc",
        "mq135_voltage",
        "mq3_voltage",
        "bme688_gas_resistance",
        "bme688_iaq",
        "bme688_breath_voc_equivalent",
        "scd40_co2",
    ]

    available_chart_cols = [c for c in chart_cols if c in df.columns]

    selected_cols = st.multiselect(
        "Sensores a visualizar",
        options=available_chart_cols,
        default=available_chart_cols[:4],
    )

    if selected_cols:
        plot_df = df[["elapsed_s", *selected_cols]].copy()
        plot_df = plot_df.set_index("elapsed_s")
        st.line_chart(plot_df)
else:
    st.info("O gráfico aparece depois de pelo menos duas leituras.")


# -----------------------------
# Debug/download
# -----------------------------

with st.expander("Dados da sessão"):
    if records:
        df_records = pd.DataFrame(records)
        st.dataframe(df_records.tail(20), use_container_width=True)

        st.download_button(
            "Descarregar sessão atual em CSV",
            data=df_records.to_csv(index=False).encode("utf-8"),
            file_name="electronic_nose_session.csv",
            mime="text/csv",
        )
    else:
        st.write("Sem dados.")


# -----------------------------
# Auto refresh
# -----------------------------

if st.session_state.running:
    time.sleep(read_interval_ms / 1000.0)
    st.rerun()
