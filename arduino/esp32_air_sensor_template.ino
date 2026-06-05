/*
  Template ESP32 para a atividade "IA que vê o ar".

  O dashboard Python espera receber linhas CSV:
  timestamp_ms,voc,pm25,temp,humidity

  Este sketch usa valores simulados no ESP32. Substitui as funções readVOC(),
  readPM25(), readTemp() e readHumidity() pelas leituras reais dos teus sensores.

  Sensores possíveis:
  - VOC: SGP30, SGP40, BME680, CCS811
  - PM2.5: PMS5003, SPS30, SDS011
  - Temp/Humidade: BME280, DHT22, SHT31
*/

void setup() {
  Serial.begin(115200);
  delay(1000);
}

float readVOC() {
  return 120 + random(-20, 20);
}

float readPM25() {
  return 4 + random(-10, 10) / 10.0;
}

float readTemp() {
  return 22.0 + random(-10, 10) / 20.0;
}

float readHumidity() {
  return 50.0 + random(-20, 20) / 5.0;
}

void loop() {
  unsigned long timestamp = millis();
  float voc = readVOC();
  float pm25 = readPM25();
  float temp = readTemp();
  float humidity = readHumidity();

  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(voc, 2);
  Serial.print(",");
  Serial.print(pm25, 2);
  Serial.print(",");
  Serial.print(temp, 2);
  Serial.print(",");
  Serial.println(humidity, 2);

  delay(1000);
}
