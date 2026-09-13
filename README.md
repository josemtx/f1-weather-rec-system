# F1-Weather-Rec

Predicción probabilística de resultados de Fórmula 1, con un dashboard estático
que dice honestamente qué acierta y qué no.

Ingesta (Java + Python → MongoDB) · 81 features sin fuga temporal · XGBoost
anclado a la parrilla · simulación Monte Carlo con intervalos calibrados ·
registro auditable de cada predicción antes de la carrera.

## Qué predice y cómo de bien

Para cada carrera: probabilidad de victoria, podio y puntos por piloto, dónde
acaba *si termina* (intervalo P10–P90) y probabilidad de abandono. Todo medido
sobre la temporada 2025 completa con un modelo que nunca la vio:

| | Error medio (puestos) | Ancla sola |
|---|---|---|
| Con clasificación (ancla: parrilla) | **2.49** | 2.88 |
| Sin clasificación (ancla: ritmo reciente) | 3.11 | 3.07 |

La lectura honesta: conocida la parrilla, el modelo la mejora en ~0.4 puestos;
antes de la clasificación no sabe más que el ritmo reciente del piloto. La
mayor parte del acierto viene de la información, no del aprendizaje
automático — y el dashboard lo muestra así. Los intervalos del 80 % cubren el
80.4 % de los resultados reales (calibrado, no supuesto).

## Cómo está hecho

```
java-app/            Ingesta OpenF1 (sesiones, vueltas, stints, boxes, sensores de pista)
                     y pronósticos OpenWeatherMap (Task Scheduler diario)
ml/jolpica/          Resultados, clasificación y calendario 2018-2026 (Jolpica/Ergast)
ml/ingestion/        Clima histórico: NASA POWER + Copernicus ERA5; catálogo de circuitos
ml/features/         92 columnas en 10 categorías, anti-fuga estructural (shift(1)),
                     filas "fantasma" para carreras futuras
ml/training/         train (evaluation + production), ablación por régimen de información,
                     calibración de intervalos, diagnóstico de residuos, ancla del regresor
ml/simulation/       Monte Carlo: clima, fiabilidad, boxes y ruido residual calibrado
ml/predictions/      Registro de predicciones en Mongo con el régimen de información
ml/dashboard/        Exportador a data.json + página estática (index.html)
ml/docs/             ARCHITECTURE.md (decisiones) · HANDOFF.md (estado y cómo retomar)
```

Base de datos: MongoDB `F1-WeatherRec-Prod` (fijada en `.env` como `MONGODB_DB`;
no hay valor por defecto a propósito).

## Puesta en marcha

Requisitos: Python 3.11+, Java 17 + Maven, MongoDB local.

```bash
python -m venv venv && ./venv/Scripts/pip install -r ml/requirements.txt
cp .env.example .env     # MONGODB_URI, MONGODB_DB, OPENWEATHER_API_KEY, CDS_URL/CDS_KEY

# Datos
cd java-app && mvn clean package && java -jar target/F1-WeatherRec.jar f1 && cd ..
python -m ml.ingestion.load_circuits
python -m ml.jolpica.ingest_jolpica
python -m ml.ingestion.ingest_climate          # NASA POWER
python -m ml.ingestion.ingest_era5             # opcional, requiere cuenta CDS

# Modelo
python -m ml.features.build_features
python -m ml.training.train
python -m ml.training.ablation
python -m ml.training.calibrate_simulation     # y `... <version> pre_quali`

# Cada fin de semana
python -m ml.predictions.predict_race          # viernes y tras la clasificación
python -m ml.dashboard.export_data             # regenera ml/dashboard/web/data.json
python -m pytest ml/tests -q
```

El dashboard es HTML autocontenido: `ml/dashboard/web/` se sirve tal cual
(GitHub Pages o `python -m http.server` en esa carpeta).

## Decisiones que conviene conocer

- **El regresor predice el delta frente a un ancla**, no la posición absoluta.
  Medido: el modelo absoluto no superaba a "predecir = parrilla". Ver
  `ml/training/anchor.py`.
- **Los abandonos no entran en el regresor**: los pone el simulador con su
  propio dado. Mezclarlos sesgaba +1 puesto a todos los demás y destrozaba los
  intervalos.
- **El clima aporta ~0.02 puestos.** Con ~5 % de carreras mojadas es un techo
  de datos, no de diseño; está documentado en el dashboard, no escondido.
- **Todo lo que se mide se mide fuera de muestra**, y las predicciones se
  registran antes de la carrera y no se pueden reescribir después.

Más detalle en [ml/docs/ARCHITECTURE.md](ml/docs/ARCHITECTURE.md).
