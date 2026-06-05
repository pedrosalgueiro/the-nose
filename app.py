from __future__ import annotations

import os
import sys
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import DEFAULT_MQTT_TOPICS, DISPLAY_COLUMNS, LABELS, SENSOR_COLUMNS, WINDOW_SIZE
from features import features_for_latest_window
from mqtt_listener import MQTTConfig, MQTTLiveReader
from simulator import live_simulator

load_dotenv()

MODEL_PATH = "models/air_ai_model.joblib"

st.set_page_config(page_title="IA que vê o ar", page_icon="🌱", layout="wide")


def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def env_topic(key: str) -> str:
    return os.getenv("MQTT_TOPIC_" + key.upper(), DEFAULT_MQTT_TOPICS[key])


def sustainable_recommendation(label: str, confidence: float, latest: dict) -> tuple[str, str]:
    voc = latest.get("voc", 0)
    nox = latest.get("nox", 0)
    mq3 = latest.get("mq3_voltage", 0)
    mq135 = latest.get("mq135_voltage", 0)
    co2 = latest.get("scd40_co2", 0)
    iaq = latest.get("bme688_iaq", 0)
    humidity = latest.get("sht40_humidity", latest.get("scd40_humidity", 0))

    if confidence < 0.45:
        return "🟡", "Padrão incerto. Continua a recolher dados antes de tomar uma decisão automática."
    if co2 > 1200:
        return "🔵", "CO₂ elevado. Ventilação controlada é recomendada; aqui a sustentabilidade vem de ventilar quando é necessário."
    if voc > 250 or mq3 > 0.9 or mq135 > 0.9 or iaq > 120:
        return "🔵", "Evento de gases/VOC detetado. Ventilação breve e localizada melhora o ar sem climatizar em excesso."
    if nox > 8:
        return "🔴", "NOx Index elevado. Investigar fonte de poluição/combustão antes de decidir a estratégia de ventilação."
    if humidity > 70:
        return "🟡", "Humidade elevada. Ventilar de forma controlada pode reduzir risco de bolor e desconforto."
    if label == "ar_limpo" and voc < 150 and co2 < 900 and mq3 < 0.8 and mq135 < 0.8:
        return "🟢", "Ar aparentemente estável. Não é necessário aumentar ventilação — poupa-se energia."
    return "🟢", "Situação aceitável. A monitorização contínua permite agir só quando faz sentido."


def predict(records: list[dict], bundle):
    X = features_for_latest_window(records, WINDOW_SIZE)
    if X is None or bundle is None:
        return None
    model = bundle["model"]
    expected = bundle.get("feature_columns")
    if expected:
        X = X.reindex(columns=expected, fill_value=0)
    labels = list(model.classes_) if hasattr(model, "classes_") else bundle.get("labels", [])
    probs = model.predict_proba(X)[0]
    best_idx = int(np.argmax(probs))
    return {"label": labels[best_idx], "confidence": float(probs[best_idx]), "probs": dict(zip(labels, probs))}


def mqtt_generator(reader: MQTTLiveReader):
    while True:
        yield reader.read()


def stop_mqtt_reader():
    reader = st.session_state.get("mqtt_reader")
    if reader is not None:
        reader.stop()
    st.session_state.mqtt_reader = None


def metric_value(value, suffix="", decimals=1):
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return "—"


def last_seen_age(latest: dict, key: str) -> str:
    seen = latest.get("mqtt_last_seen", {}).get(key)
    if not seen:
        return "sem dados"
    age = time.time() - seen
    return f"há {age:.0f}s"


st.title("🌱 IA que vê o ar")
st.caption("Dashboard para ESPHome + MQTT: SGP41, SHT40, BME688, SCD40, ADS1115, MQ-135 e MQ-3.")

with st.sidebar:
    st.header("Configuração")
    mode = st.radio("Modo", ["MQTT / ESPHome", "Simulado"], index=0)
    event = st.selectbox("Evento simulado", LABELS, index=0)
    max_points = st.slider("Pontos no gráfico", 60, 600, 180, step=20)
    refresh_ms = st.slider("Intervalo de leitura", 500, 5000, 1000, step=250)
    model_path = st.text_input("Modelo", MODEL_PATH)

    topic_map = {}
    if mode == "MQTT / ESPHome":
        st.subheader("MQTT")
        mqtt_broker = st.text_input("Broker", os.getenv("MQTT_BROKER", "10.0.0.2"))
        mqtt_port = st.number_input("Porta", value=int(os.getenv("MQTT_PORT", "1883")), step=1)
        mqtt_username = st.text_input("Utilizador", os.getenv("MQTT_USERNAME", ""))
        mqtt_password = st.text_input("Password", os.getenv("MQTT_PASSWORD", ""), type="password")

        with st.expander("Tópicos MQTT do voc-sensor.yaml", expanded=False):
            for key in SENSOR_COLUMNS:
                label = DISPLAY_COLUMNS.get(key, key)
                topic_map[key] = st.text_input(label, env_topic(key))
    run = st.toggle("Iniciar demonstração", value=False)
    st.divider()
    st.markdown("**Protocolo de previsão/treino:** 30 s ar limpo → 60 s amostra → 90 s recuperação. A previsão fica mais fiável após ~180 s.")

bundle = load_model(model_path)
if bundle is None:
    st.warning("Modelo não encontrado. Podes usar a app para ver sensores, mas ainda não há previsão IA.")

if "records" not in st.session_state:
    st.session_state.records = deque(maxlen=1200)
if "generator_key" not in st.session_state:
    st.session_state.generator_key = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "mqtt_reader" not in st.session_state:
    st.session_state.mqtt_reader = None

if mode == "MQTT / ESPHome":
    key = (mode, mqtt_broker, int(mqtt_port), mqtt_username, tuple(sorted(topic_map.items())))
else:
    key = (mode, event)

if st.session_state.generator_key != key:
    st.session_state.records.clear()
    st.session_state.generator_key = key
    st.session_state.generator = None
    stop_mqtt_reader()
    if mode == "Simulado":
        st.session_state.generator = live_simulator(event)

if run:
    try:
        if mode == "MQTT / ESPHome" and st.session_state.generator is None:
            cfg = MQTTConfig(
                broker=mqtt_broker,
                port=int(mqtt_port),
                username=mqtt_username,
                password=mqtt_password,
                topics=topic_map or dict(DEFAULT_MQTT_TOPICS),
            )
            reader = MQTTLiveReader(cfg)
            reader.start()
            st.session_state.mqtt_reader = reader
            st.session_state.generator = mqtt_generator(reader)

        row = next(st.session_state.generator)
        for col in SENSOR_COLUMNS:
            row.setdefault(col, 0.0)
        st.session_state.records.append(row)
    except Exception as exc:
        st.error(f"Erro de leitura: {exc}")

records = list(st.session_state.records)

if not records:
    st.markdown("## Carrega em **Iniciar demonstração** para começar.")
    st.stop()

df = pd.DataFrame(records)
df_plot = df.tail(max_points).copy()
df_plot["t"] = np.arange(len(df_plot))
latest = records[-1]
prediction = predict(records, bundle)

if mode == "MQTT / ESPHome":
    status = "ligado" if latest.get("mqtt_connected") else "a ligar/desligado"
    st.caption(f"Estado MQTT: **{status}**")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("VOC Index", metric_value(latest.get("voc"), decimals=0))
m2.metric("NOx Index", metric_value(latest.get("nox"), decimals=0))
m3.metric("SCD40 CO₂", metric_value(latest.get("scd40_co2"), " ppm", 0))
m4.metric("MQ-135", metric_value(latest.get("mq135_voltage"), " V", 3))
m5.metric("MQ-3", metric_value(latest.get("mq3_voltage"), " V", 3))
m6.metric("BME688 IAQ", metric_value(latest.get("bme688_iaq"), decimals=0))

m7, m8, m9, m10, m11, m12 = st.columns(6)
m7.metric("SHT40 Temp", metric_value(latest.get("sht40_temperature"), " °C", 1))
m8.metric("SHT40 Hum", metric_value(latest.get("sht40_humidity"), "%", 0))
m9.metric("BME Gas R", metric_value(latest.get("bme688_gas_resistance"), " Ω", 0))
m10.metric("BME bVOC", metric_value(latest.get("bme688_breath_voc_equivalent"), " ppm", 2))
m11.metric("BME eCO₂", metric_value(latest.get("bme688_co2_equivalent"), " ppm", 0))
m12.metric("BME Press", metric_value(latest.get("bme688_pressure"), " hPa", 1))

left, right = st.columns([2, 1])
with left:
    chart_groups = {
        "Gases / IAQ": ["voc", "nox", "bme688_iaq", "bme688_breath_voc_equivalent"],
        "Analógicos MQ": ["mq135_voltage", "mq3_voltage"],
        "CO₂": ["scd40_co2", "bme688_co2_equivalent"],
        "Ambiente": ["sht40_temperature", "sht40_humidity", "bme688_temperature", "bme688_humidity", "bme688_pressure"],
        "BME688 Gas Resistance": ["bme688_gas_resistance"],
    }
    selected_group = st.selectbox("Grupo de gráfico", list(chart_groups.keys()))
    cols = [c for c in chart_groups[selected_group] if c in df_plot.columns]
    long = df_plot.melt(id_vars="t", value_vars=cols, var_name="sensor", value_name="valor")
    long["sensor"] = long["sensor"].map(DISPLAY_COLUMNS).fillna(long["sensor"])
    fig = px.line(long, x="t", y="valor", color="sensor", markers=False, title=selected_group)
    fig.update_layout(height=430, legend_title_text="Sensor")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Previsão IA")
    if prediction is None:
        st.info(f"A aguardar dados suficientes. Para o modelo segmentado, o ideal são {WINDOW_SIZE} amostras (~180 s).")
    else:
        label = prediction["label"]
        conf = prediction["confidence"]
        st.metric("Classe prevista", label)
        st.progress(conf)
        st.caption(f"Confiança: {conf:.0%}")
        emoji, rec = sustainable_recommendation(label, conf, latest)
        st.markdown(f"### {emoji} Decisão sustentável")
        st.info(rec)
        probs_df = pd.DataFrame({"classe": list(prediction["probs"].keys()), "probabilidade": list(prediction["probs"].values())})
        st.dataframe(probs_df.sort_values("probabilidade", ascending=False), hide_index=True, use_container_width=True)

with st.expander("Estado dos tópicos MQTT / últimas mensagens", expanded=False):
    if mode == "MQTT / ESPHome":
        rows = []
        for key in SENSOR_COLUMNS:
            rows.append({
                "sensor": DISPLAY_COLUMNS.get(key, key),
                "tópico": topic_map.get(key, DEFAULT_MQTT_TOPICS.get(key, "")),
                "valor atual": latest.get(key),
                "última mensagem": last_seen_age(latest, key),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.write("Disponível apenas em modo MQTT.")

with st.expander("Dados recentes", expanded=False):
    show_cols = ["timestamp_ms"] + SENSOR_COLUMNS
    st.dataframe(df.tail(50)[[c for c in show_cols if c in df.columns]], use_container_width=True)

if run:
    time.sleep(refresh_ms / 1000)
    st.rerun()
