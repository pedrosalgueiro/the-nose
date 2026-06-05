# IA que vê o ar — Dashboard para nariz eletrónico ESPHome/MQTT

Esta versão está adaptada ao teu `electronic-nose.yaml`:

- ESP32 com ESPHome;
- MQTT no broker `10.0.0.2:1883` por defeito;
- dois barramentos I2C;
- SGP41, SHT40, ADS1115, BME688, SCD40;
- MQ-135 no ADS1115 A0;
- MQ-3 no ADS1115 A1.

## Instalação

```bash
cd ia_ar_sustentavel_voc_sensor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

No Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tópicos MQTT esperados

A aplicação já vem configurada para estes tópicos:

```text
electronic-nose/sensor/voc_index/state
electronic-nose/sensor/nox_index/state
electronic-nose/sensor/mq135_voltage/state
electronic-nose/sensor/mq3_voltage/state
electronic-nose/sensor/sht40_temperature/state
electronic-nose/sensor/sht40_humidity/state
electronic-nose/sensor/bme688_temperature/state
electronic-nose/sensor/bme688_humidity/state
electronic-nose/sensor/bme688_pressure/state
electronic-nose/sensor/bme688_gas_resistance/state
electronic-nose/sensor/bme688_iaq/state
electronic-nose/sensor/bme688_co2_equivalent/state
electronic-nose/sensor/bme688_breath_voc_equivalent/state
electronic-nose/sensor/scd40_co2/state
electronic-nose/sensor/scd40_temperature/state
electronic-nose/sensor/scd40_humidity/state
```

Podes alterar todos estes tópicos no painel lateral da app ou copiando `.env.example` para `.env`.

## Recolher dados reais

Exemplo:

```bash
python scripts/collect_mqtt_data.py --label cafe --session cafe_001 --duration 150 --broker 10.0.0.2
```

Protocolo recomendado:

```text
0-30 s      ar limpo / referência
30-75 s     aproximar a amostra
75-150 s    retirar e deixar recuperar
```

Repete várias sessões por classe:

```bash
python scripts/collect_mqtt_data.py --label ar_limpo --session ar_limpo_001 --duration 150 --broker 10.0.0.2
python scripts/collect_mqtt_data.py --label cafe --session cafe_001 --duration 150 --broker 10.0.0.2
python scripts/collect_mqtt_data.py --label perfume --session perfume_001 --duration 150 --broker 10.0.0.2
python scripts/collect_mqtt_data.py --label alcool --session alcool_001 --duration 150 --broker 10.0.0.2
python scripts/collect_mqtt_data.py --label vinagre --session vinagre_001 --duration 150 --broker 10.0.0.2
```

Os dados ficam em:

```text
data/real_readings.csv
```

## Treinar o modelo com dados reais

```bash
python src/train_model.py --input data/real_readings.csv --output models/air_ai_model.joblib
```

Depois reinicia a app Streamlit.

## Nota importante

O modelo incluído é apenas uma demonstração sintética para testar a aplicação. Para previsões reais na feira, recolhe dados reais com o teu sensor, no teu ambiente e com o mesmo protocolo experimental.

## Treino com protocolo baseline → exposição → recuperação

Esta versão do projeto treina o modelo com **uma linha de features por sessão completa**. O treino já não usa janelas deslizantes genéricas como exemplos independentes.

Protocolo assumido por defeito:

```text
0–30 s      baseline / ar limpo
30–90 s     exposição à amostra
90–180 s    recuperação
```

O `src/features.py` calcula, para cada sensor, features como:

- média e desvio na baseline;
- média, pico, variação e área durante a exposição;
- declive de subida;
- recuperação face à baseline;
- valor final menos baseline;
- tempo até ao pico.

Isto significa que podes reaproveitar os CSV já recolhidos, desde que tenham as colunas:

```text
elapsed_s,label,session_id
```

Para treinar com dados reais:

```bash
python src/train_model.py \
  --input data/real_readings.csv \
  --output models/air_ai_model.joblib
```

A app passa a esperar idealmente cerca de **180 segundos** de dados para uma previsão completa, porque o modelo foi treinado com a sessão inteira.
