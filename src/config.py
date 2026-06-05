LABELS = [
    "ar_limpo",
    "cafe",
    "perfume",
    "alcool",
    "vinagre",
    "particulas",
]

# Mapa normalizado usado pela aplicação/modelo.
# As colunas MQTT do ESPHome são convertidas para estes nomes curtos.
SENSOR_COLUMNS = [
    "voc",                 # SGP41 VOC Index
    "nox",                 # SGP41 NOx Index
    "mq135_voltage",       # ADS1115 A0
    "mq3_voltage",         # ADS1115 A1
    "sht40_temperature",
    "sht40_humidity",
    "bme688_temperature",
    "bme688_humidity",
    "bme688_pressure",
    "bme688_gas_resistance",
    "bme688_iaq",
    "bme688_co2_equivalent",
    "bme688_breath_voc_equivalent",
    "scd40_co2",
    "scd40_temperature",
    "scd40_humidity",
]

DISPLAY_COLUMNS = {
    "voc": "SGP41 VOC Index",
    "nox": "SGP41 NOx Index",
    "mq135_voltage": "MQ-135 Voltage",
    "mq3_voltage": "MQ-3 Voltage",
    "sht40_temperature": "SHT40 Temp",
    "sht40_humidity": "SHT40 Hum",
    "bme688_temperature": "BME688 Temp",
    "bme688_humidity": "BME688 Hum",
    "bme688_pressure": "BME688 Press",
    "bme688_gas_resistance": "BME688 Gas R",
    "bme688_iaq": "BME688 IAQ",
    "bme688_co2_equivalent": "BME688 eCO2",
    "bme688_breath_voc_equivalent": "BME688 bVOC",
    "scd40_co2": "SCD40 CO2",
    "scd40_temperature": "SCD40 Temp",
    "scd40_humidity": "SCD40 Hum",
}

# Defaults baseados no teu electronic-nose.yaml / ESPHome.
DEFAULT_MQTT_TOPICS = {
    "voc": "electronic-nose/sensor/voc_index/state",
    "nox": "electronic-nose/sensor/nox_index/state",
    "mq135_voltage": "electronic-nose/sensor/mq135_voltage/state",
    "mq3_voltage": "electronic-nose/sensor/mq3_voltage/state",
    "sht40_temperature": "electronic-nose/sensor/sht40_temperature/state",
    "sht40_humidity": "electronic-nose/sensor/sht40_humidity/state",
    "bme688_temperature": "electronic-nose/sensor/bme688_temperature/state",
    "bme688_humidity": "electronic-nose/sensor/bme688_humidity/state",
    "bme688_pressure": "electronic-nose/sensor/bme688_pressure/state",
    "bme688_gas_resistance": "electronic-nose/sensor/bme688_gas_resistance/state",
    "bme688_iaq": "electronic-nose/sensor/bme688_iaq/state",
    "bme688_co2_equivalent": "electronic-nose/sensor/bme688_co2_equivalent/state",
    "bme688_breath_voc_equivalent": "electronic-nose/sensor/bme688_breath_voc_equivalent/state",
    "scd40_co2": "electronic-nose/sensor/scd40_co2/state",
    "scd40_temperature": "electronic-nose/sensor/scd40_temperature/state",
    "scd40_humidity": "electronic-nose/sensor/scd40_humidity/state",
}

WINDOW_SIZE = 180
STEP_SIZE = 5
RANDOM_STATE = 42
