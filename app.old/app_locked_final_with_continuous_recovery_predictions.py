from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

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

DEFAULT_FULL_MODEL_PATH = "models/electronic_nose_indoor_outdoor.joblib"
DEFAULT_FAST_MODEL_PATH = "models/electronic_nose_fast.joblib"

FULL_WINDOW_SIZE = 180
SESSION_END_S = 180.0
RECOVERY_STABLE_SECONDS = 30.0

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

    This is only for continuous diagnostic classification after the final
    session result is locked. It prevents elapsed_s > 180 from breaking the
    segmented feature extractor.
    """
    window = list(records[-window_size:])
    normalized: list[dict[str, Any]] = []

    for idx, row in enumerate(window):
        new_row = dict(row)
        new_row["elapsed_s"] = float(idx)
        normalized.append(new_row)

    return normalized


def predict_from_records(
    records: list[dict[str, Any]],
    bundle: dict[str, Any] | None,
    mode: str,
    *,
    normalize_window: bool = False,
) -> dict[str, Any] | None:
    if bundle is None:
        return None

    if mode == "fast":
        if normalize_window:
            records = normalize_window_records(records, 70)
        X = features_for_fast_window(records)
    else:
        if normalize_window:
            records = normalize_window_records(records, FULL_WINDOW_SIZE)
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
    if not records:
        return {}

    df = pd.DataFrame(records).copy()
    if "elapsed_s" not in df.columns:
        return {}

    baseline_df = df[(df["elapsed_s"] >= 0) & (df["elapsed_s"] < 25)].copy()
    if baseline_df.empty:
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


def reset_measurement() -> None:
    now = time.time()
    st.session_state.session_records = []
    st.session_state.monitor_records = []
    st.session_state.continuous_prediction_history = []
    st.session_state.measurement_start_time = now
    st.session_state.running = True
    st.session_state.session_finished = False
    st.session_state.final_prediction_locked = None
    st.session_state.fast_prediction_locked = None
    st.session_state.baseline_locked = {}


def stop_measurement() -> None:
    st.session_state.running = False
    # If the session was already completed, keep monitoring disabled too.
    st.session_state.session_finished = False


def status_text(elapsed_s: float, finished: bool) -> str:
    if finished:
        return "Sessão concluída. Previsão final congelada; monitorização contínua ativa."
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
    return "Sessão completa: a previsão final será congelada."


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

defaults = {
    "session_records": [],
    "monitor_records": [],
    "continuous_prediction_history": [],
    "measurement_start_time": time.time(),
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
# Sidebar
# -------------------------------------------------------------------

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

full_model_path = st.sidebar.text_input("Modelo completo", value=DEFAULT_FULL_MODEL_PATH)
fast_model_path = st.sidebar.text_input("Modelo rápido", value=DEFAULT_FAST_MODEL_PATH)

with st.sidebar.expander("Tópicos MQTT"):
    for key, topic in DEFAULT_MQTT_TOPICS.items():
        st.code(f"{key}: {topic}", language="text")

with st.sidebar.expander("Monitorização da recuperação"):
    st.write("Não usa recovery sessions/classes. Usa comparação com a baseline inicial.")
    st.json(READY_THRESHOLDS)


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

st.title("👃 Electronic Nose")
st.caption("IA + sensores ambientais para reconhecer padrões de odor.")

col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    if st.button("Iniciar nova medição", type="primary", use_container_width=True):
        reset_measurement()

with col_b:
    if st.button("Parar leitura", use_container_width=True):
        stop_measurement()

with col_c:
    st.info(
        "Protocolo: 0–25 s ar limpo, 25–35 s aproximar amostra, "
        "35–90 s exposição, 90–180 s recuperação. Aos 180 s a previsão final congela."
    )


# -------------------------------------------------------------------
# Read one sample from MQTT
# -------------------------------------------------------------------

should_read = st.session_state.running or st.session_state.session_finished

if should_read:
    latest = reader.read()
    elapsed_s = time.time() - st.session_state.measurement_start_time

    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "elapsed_s": elapsed_s,
    }
    for col in SENSOR_COLUMNS:
        row[col] = latest.get(col)

    # Continua depois dos 180 s para saber quando volta a clean air.
    st.session_state.monitor_records.append(row)
    st.session_state.monitor_records = st.session_state.monitor_records[-1200:]

    # A sessão usada para previsão final congela aos 180 s.
    if st.session_state.running and elapsed_s <= SESSION_END_S:
        st.session_state.session_records.append(row)
        st.session_state.session_records = st.session_state.session_records[-600:]


session_records = st.session_state.session_records
monitor_records = st.session_state.monitor_records

elapsed = 0.0
if session_records:
    elapsed = float(session_records[-1].get("elapsed_s", 0.0))
elif monitor_records:
    elapsed = float(monitor_records[-1].get("elapsed_s", 0.0))


# -------------------------------------------------------------------
# Predictions and automatic final lock
# -------------------------------------------------------------------

fast_prediction = predict_from_records(session_records, fast_bundle, mode="fast")

# Important: do NOT calculate/show the final prediction before 180 s.
# This avoids confidence values changing before the official final result.
full_prediction_for_lock = None

if fast_prediction is not None:
    st.session_state.fast_prediction_locked = fast_prediction

if not st.session_state.baseline_locked and len(session_records) >= 25:
    baseline = compute_baseline(session_records)
    if baseline:
        st.session_state.baseline_locked = baseline

if st.session_state.running and elapsed >= SESSION_END_S:
    full_prediction_for_lock = predict_from_records(session_records, full_bundle, mode="full")

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
if st.session_state.session_finished and len(monitor_records) >= FULL_WINDOW_SIZE:
    # Diagnostic rolling classification, not the official final result.
    continuous_prediction = predict_from_records(
        monitor_records,
        full_bundle,
        mode="full",
        normalize_window=True,
    )

    if continuous_prediction is not None:
        history_row = {
            "elapsed_s": float(monitor_records[-1].get("elapsed_s", 0.0)),
            "label": continuous_prediction["label"],
            "confidence": continuous_prediction["confidence"],
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
# Status
# -------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Tempo da sessão", f"{min(elapsed, SESSION_END_S):.0f} s")

with m2:
    st.metric("Leituras da sessão", len(session_records))

with m3:
    st.metric("Previsão rápida", "Pronta" if display_fast_prediction is not None else "A aguardar")

with m4:
    st.metric(
        "Previsão final",
        "Congelada" if display_full_prediction is not None else "A aguardar 180 s",
    )

st.progress(min(elapsed / SESSION_END_S, 1.0))
st.write(f"**Estado:** {status_text(elapsed, st.session_state.session_finished)}")

if display_full_prediction is not None:
    st.success("Sessão concluída: a previsão final foi congelada aos 180 s.")


# -------------------------------------------------------------------
# Predictions UI
# -------------------------------------------------------------------

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.subheader("Previsão rápida (~60 s)")
    if fast_bundle is None:
        st.warning(f"Modelo rápido não encontrado: `{fast_model_path}`")
    elif display_fast_prediction is None:
        st.info("A aguardar dados suficientes para previsão rápida.")
    else:
        st.metric(
            "Resultado preliminar",
            display_fast_prediction["label"],
            f"{display_fast_prediction['confidence'] * 100:.1f}% confiança",
        )
        st.bar_chart(pd.Series(display_fast_prediction["probs"]).sort_values(ascending=False))

with pred_col2:
    st.subheader("Previsão final — congelada aos 180 s")
    if full_bundle is None:
        st.warning(f"Modelo completo não encontrado: `{full_model_path}`")
    elif display_full_prediction is None:
        st.info("A previsão final só será calculada aos 180 s.")
    else:
        st.metric(
            "Resultado final",
            display_full_prediction["label"],
            f"{display_full_prediction['confidence'] * 100:.1f}% confiança",
        )
        st.caption("Este resultado já não muda. Foi congelado aos 180 s.")
        st.bar_chart(pd.Series(display_full_prediction["probs"]).sort_values(ascending=False))


# -------------------------------------------------------------------
# Continuous recovery classification UI
# -------------------------------------------------------------------

st.subheader("Classificação contínua durante recuperação")

if not st.session_state.session_finished:
    st.info("A classificação contínua aparece depois de a previsão final ficar congelada aos 180 s.")
elif continuous_prediction is None:
    st.info("A aguardar dados suficientes para classificação contínua.")
else:
    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric(
            "Classificação atual",
            continuous_prediction["label"],
            f"{continuous_prediction['confidence'] * 100:.1f}% confiança",
        )
        st.caption("Diagnóstico contínuo: não substitui a previsão final congelada.")

    with c2:
        st.bar_chart(pd.Series(continuous_prediction["probs"]).sort_values(ascending=False))

    history = st.session_state.continuous_prediction_history
    if len(history) >= 2:
        history_df = pd.DataFrame(history)
        prob_cols = [
            col for col in history_df.columns
            if col not in {"elapsed_s", "label", "confidence"}
        ]
        if prob_cols:
            st.write("Evolução das probabilidades durante a recuperação")
            st.line_chart(history_df.set_index("elapsed_s")[prob_cols])


# -------------------------------------------------------------------
# Continuous readiness monitor
# -------------------------------------------------------------------

st.subheader("Estado da câmara para nova amostra")

ready, ready_info = is_ready_for_next_sample(
    monitor_records=monitor_records,
    baseline=st.session_state.baseline_locked,
)

ready_col1, ready_col2 = st.columns([1, 2])

with ready_col1:
    if ready:
        st.success("Pronta / clean air")
        st.metric("Pode testar nova amostra?", "Sim")
    else:
        st.warning(ready_info.get("state", "A recuperar"))
        st.metric("Pode testar nova amostra?", "Não")

with ready_col2:
    st.write(ready_info.get("reason", ""))

    checks = ready_info.get("checks", {})
    if checks:
        checks_df = pd.DataFrame(
            [{"critério": key, "ok": bool(value)} for key, value in checks.items()]
        )
        st.dataframe(checks_df, use_container_width=True, hide_index=True)

    details = ready_info.get("details", {})
    if details:
        with st.expander("Detalhes dos desvios face à baseline"):
            st.json({k: round(float(v), 5) for k, v in details.items()})


# -------------------------------------------------------------------
# Latest sensor readings
# -------------------------------------------------------------------

st.subheader("Leituras atuais dos sensores")

if monitor_records:
    latest_record = monitor_records[-1]
    sensor_df = pd.DataFrame(
        [{"sensor": col, "value": latest_record.get(col)} for col in SENSOR_COLUMNS]
    )
    st.dataframe(sensor_df, use_container_width=True, hide_index=True)
else:
    st.info("Ainda não há leituras.")


# -------------------------------------------------------------------
# Charts
# -------------------------------------------------------------------

st.subheader("Histórico da sessão")

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
        "Sensores a visualizar",
        options=available_chart_cols,
        default=available_chart_cols[:4],
    )

    if selected_cols:
        st.line_chart(df_session.set_index("elapsed_s")[selected_cols])
else:
    st.info("O gráfico aparece depois de pelo menos duas leituras.")

with st.expander("Histórico contínuo / recuperação"):
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
            st.line_chart(df_monitor.set_index("elapsed_s")[available_recovery_cols])
    else:
        st.write("Sem dados contínuos suficientes.")


# -------------------------------------------------------------------
# Data/debug
# -------------------------------------------------------------------

with st.expander("Dados"):
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

    if st.session_state.continuous_prediction_history:
        st.write("Histórico de classificação contínua")
        st.dataframe(pd.DataFrame(st.session_state.continuous_prediction_history).tail(20), use_container_width=True)


# -------------------------------------------------------------------
# Auto refresh
# -------------------------------------------------------------------

if st.session_state.running or st.session_state.session_finished:
    time.sleep(read_interval_ms / 1000.0)
    st.rerun()
