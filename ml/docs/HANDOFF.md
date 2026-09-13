# HANDOFF — estado del proyecto y cómo retomarlo

Última actualización: **2026-09-13, ~23:20 UTC** (domingo, tras el GP de Madrid y la limpieza del repo).
Escrito para que la siguiente sesión arranque sin releer todo el historial.

---

## 1. Qué es esto en una frase

Sistema predictivo de resultados de F1: ingesta (Java+Python → MongoDB `F1-WeatherRec-Prod`),
80 features sin fuga temporal, XGBoost (4 targets), simulación Monte Carlo calibrada,
registro auditable de predicciones, y un dashboard estático publicable en GitHub Pages.
Objetivo: proyecto insignia de portfolio en GitHub, con **honestidad sobre lo que acierta y lo que no**.

Base de datos: **siempre `F1-WeatherRec-Prod`**. Nunca `F1-WeatherRec` (la vieja; hubo un
incidente escribiendo ahí por error — ver ARCHITECTURE.md). Fijar `MONGO_DB=F1-WeatherRec-Prod`
en cada comando Java; Python lo lee de `.env`.

---

## 2. Lo que está hecho y funciona (commiteado, 10 commits)

| Capa | Estado |
|---|---|
| Ingesta OpenF1 (Java) | 4 tipos de sesión (Race/Qualifying/Sprint/Sprint Qualifying), 2023-2026, laps/stints/pit/weather/positions. Backoff exponencial. |
| Ingesta Jolpica (Python) | Resultados, clasificación, sprint, calendario con **horas** de sesión. 2018-2026. `--schedule-only` para refrescar solo calendario. |
| Clima | NASA POWER (año completo) + ERA5 (días de carrera) + sensores de pista OpenF1 (preferente 2023+) + pronóstico OpenWeatherMap por coordenadas de circuito (33 circuitos, 40 tramos/3h, refresco diario automático a las 06:00 vía Task Scheduler — **funciona solo, verificado**). |
| Features | 92 columnas / 80 entrenables, 10 categorías (A-J). Anti-fuga estructural (`shift(1)`), tests. Filas "fantasma" para carreras futuras (`ml/features/upcoming.py`). Temperatura de asfalto estimada por circuito para futuras. |
| Modelos | Dos con roles distintos en `ml/models/registry.json`: `evaluation` (train 2023-24, val 2025 — métricas creíbles) y `production` (2023-25 — el que predice). MAE 3.28, AUC podio 0.948. Categorías no vistas (novatos, circuitos nuevos) → NaN sin fallar. |
| Predicciones | `ml/predictions/predict_race.py` genera y **persiste** en Mongo `predictions` con régimen (pre_quali/post_sprint/post_quali) deducido automáticamente. Se niega a escribir sobre carreras ya disputadas (integridad del track record). |

---

## 3. Estado de git

**Todo commiteado** (21 commits en `main`, sin subir a remoto). Los del 13/09 por capas: regresor anclado,
simulador (régimen + relleno de parrilla), herramientas de diagnóstico, datos del dashboard, docs, y
"modelo vs parrilla" en Honestidad. Tests 24/24. Los últimos del 12/09:

```
358ddda Add session handoff document
224a136 Add shelved tool to trace a circuit from OpenF1 GPS telemetry
2a3ed49 Add static dashboard: FIA timing-screen aesthetic, publishable on GitHub Pages
6280b2d Calibrate simulation intervals and separate DNF from finishing range
```

Ficheros clave de esos commits, por si hay que tocarlos:
- `ml/simulation/monte_carlo.py` — `RESIDUAL_SCALE=0.5`, rangos "si termina", fix identidad de pilotos
- `ml/training/calibrate_simulation.py` — verifica cobertura 80% del intervalo P10-P90
- `ml/tests/test_simulator_identity.py` — driver_code no se pierde al anular categorías no vistas
- `ml/dashboard/{export_data,nerd_data,circuit_art}.py` + `ml/dashboard/web/` — dashboard estático
- `svg/` — 25 siluetas oficiales F1.com aportadas por el usuario

Tests: 24/24 en verde (`./venv/Scripts/python.exe -m pytest ml/tests/ -q`).
`git push` no se ha hecho nunca en esta sesión — decisión del usuario.

---

## 4. EL EXPERIMENTO DE MADRID — CERRADO (análisis en §4b)

Compara tres predicciones de Madrid contra la realidad:

| Momento | Estado |
|---|---|
| **Pre-clasificación** | ✅ Registrada 12/09 09:33 UTC (régimen `pre_quali`, modelo `2026-09-12_production`, clima pronóstico). VER 22.5% victoria. |
| **Post-clasificación** | ✅ Registrada 12/09 21:41 UTC (mismo modelo, régimen `post_quali`). VER 53% victoria, ANT 12%. |
| **Resultado real** | ✅ ANT gana desde P2, VER 2º, NOR 3º. Leído de OpenF1 para el análisis; ingesta Jolpica pendiente (caído). |

### Cómo ejecutar el paso post-clasificación (cuando Jolpica la tenga)

```bash
# 1. Comprobar si ya está:
curl -s "https://api.jolpi.ca/ergast/f1/2026/14/qualifying.json" | python -c "import json,sys; t=sys.stdin.read(); r=json.loads(t)['MRData']['RaceTable']['Races'] if t.strip() else None; print('SI' if r else ('AUN NO' if r==[] else 'JOLPICA SIN RESPUESTA (429/vacio), reintentar'))"

# 2. Ingerir solo 2026 (idempotente, ~2 min):
./venv/Scripts/python.exe -m ml.jolpica.ingest_jolpica 2026

# 3. Registrar la predicción — el régimen pasa SOLO a post_quali al detectar qualifying_results:
./venv/Scripts/python.exe -m ml.predictions.predict_race

# 4. Regenerar y republicar el dashboard:
./venv/Scripts/python.exe -m ml.dashboard.export_data
# → luego Artifact tool con ml/dashboard/web/index.html + files {"data.json": ...}
#   URL existente: https://claude.ai/code/artifact/fc54c822-cfad-4491-947e-3e44d2243699
```

### Domingo tras la carrera
```bash
./venv/Scripts/python.exe -m ml.jolpica.ingest_jolpica 2026   # trae el resultado
./venv/Scripts/python.exe -m ml.dashboard.export_data          # track_record se resuelve solo (join al leer)
```
La pestaña Honestidad mostrará pre vs post vs real. La ablación ya predice la diferencia esperada: ~0.5 puestos de MAE.

**Nota**: OpenF1 SÍ tiene ya la sesión de clasificación (session_key 11365, con vueltas), pero
por diseño la fuente canónica de grid/quali es Jolpica (consistencia 2018-2026). No mezclar.

**Estado a las 22:45 UTC del sábado**: Jolpica lleva caído desde ~17:00 (timeout de conexión a sus
IPs de Cloudflare; el usuario lo confirmó desde el navegador). Hay un cron de sesión cada 20 min
que ejecuta los pasos 1-4 solo cuando responda con `Races` no vacío. Si la sesión se cerró, hay que
relanzar el chequeo a mano.

---

## 4b. Regresor anclado — ACTIVADO el 13/09 tras cerrar el experimento

**Estado 13/09 ~22:40 UTC**: el experimento está cerrado (resultado real leído de OpenF1 para el análisis;
la ingesta canónica por Jolpica sigue pendiente porque volvió a caerse — un cron de sesión la espera para
cerrar el track record y republicar). El registry apunta ya a `2026-09-13_{modelB,production}` (desde el 13/09 noche las versiones se llaman `_evaluation` / `_production`) (receta
anclada); ablación y calibración regeneradas con ella. Lo que sigue es el diseño y las cifras originales.

**Resultado de Madrid** (18 finalizadores; ANT ganó desde P2, VER 2º, NOR 3º, HAM DNF en la vuelta 6):
parrilla sola MAE 1.35 y podio 3/3 (carrera muy ordenada); pre-quali registrada 4.21 (2/3); post-quali
registrada 2.32 (3/3, favorito VER 53% — ANT 12%). Con el arreglo de parrilla de abajo, el modelo nuevo
habría dado 2.03 y podio 3/3 con NOR favorito.

**Arreglo importante encontrado en el análisis** (`anchor.fill_grid_from_quali`, aplicado en el simulador):
en una carrera futura `grid_position` llega en NaN (la parrilla oficial no existe hasta el domingo) y los
modelos nunca vieron ese NaN → una predicción post-clasificación hecha en vivo rendía como pre-clasificación
(3.79 sobre 2025 vs 2.49 del backtest). Rellenar con la posición de clasificación devuelve 2.46. Afectaba
también al modelo viejo y a los clasificadores. Test en `ml/tests/test_anchor.py`.

**Medido y descartado el 13/09**: quitar `driver_code`/`constructor_id` (la identidad no ayuda ni perjudica,
±0.03) — la infravaloración de ANT no viene de ahí sino del NaN de parrilla y de su P15 en la ventana de ritmo.

Mientras Jolpica estaba caído se diagnosticó el error del modelo (`ml/training/residual_diagnosis.py`)
y salió lo importante: **el regresor no superaba a "predecir = parrilla"** (2024: 3.47 vs 2.91;
2025: 3.36 vs 3.34). La nueva receta (`ml/training/anchor.py` + `train.py`) predice el delta frente a
un ancla, solo con finalizadores y con objetivo MAE; dos regresores, uno por régimen:

| Régimen | Ancla | MAE finalizadores 2025: viejo → nuevo | Ancla sola |
|---|---|---|---|
| post_quali | parrilla (o posición de clasificación) | 2.89 → **2.48** | 2.88 |
| pre_quali | `driver_avg_finish_when_finished_last5` (ritmo, solo carreras corridas enteras) | 3.41 → **3.11** | 3.07 |

Pre-quali el modelo **empata con su ancla**: sin parrilla no sabe más que el ritmo reciente. Es honesto.
Nueva feature en `driver_form.py` (93 columnas / 81 entrenables). `RESIDUAL_SCALE` es ahora por régimen
`{post: 0.5, pre: 0.8}`, ambos medidos (cobertura 80.4% / 81.0%).

**Pendiente cuando Jolpica vuelva** (lo hace el cron de sesión, o a mano):

```bash
./venv/Scripts/python.exe -m ml.jolpica.ingest_jolpica 2026        # resultado real de Madrid -> cierra el track record
./venv/Scripts/python.exe -m ml.dashboard.export_data               # next_race pasa a la ronda 15, ya con el modelo nuevo
# republicar el artefacto fc54c822-cfad-4491-947e-3e44d2243699 (index.html + data.json)
```

Pendiente en el dashboard tras el cambio: mostrar el **MAE frente al ancla** ("el modelo vs. simplemente
la parrilla") en Honestidad — `metrics.json` ya trae `mae_ancla_val_2025` y el bloque `finish_position_pre_quali`,
`export_data.build_model_payload` aún no los exporta. Va con la ronda de feedback de la v6.

---

## 5. Dashboard estático — estado y dirección

**Fichero**: `ml/dashboard/web/index.html` (autocontenido) + `data.json` (305 KB, generado por `export_data.py`).
**Publicado como artefacto** (v9, 13/09): https://claude.ai/code/artifact/fc54c822-cfad-4491-947e-3e44d2243699
**Objetivo final**: GitHub Pages, escaparate para visitantes del repo.

**Dirección estética (decidida, no reabrir)**: pantalla de cronometraje FIA, no gráficos de TV.
Tema **único oscuro** (el usuario pidió quitar el toggle claro/oscuro). Asfalto (ruido fractal SVG),
fibra de carbono (sarga 2×2 con gradientes), bandera de cuadros, rojo F1 `#e10600` con resplandor
radial simulando foco exterior. Tipografía: **Titillium Web** (familia de la que deriva la fuente
oficial de F1) + **JetBrains Mono** para cifras. Paleta validada con el script del skill dataviz
sobre superficie oscura: rojo `#e10600` / verde `#1fa85c` / morado `#b44cff`.

**Pestañas**: Próxima carrera (héroe con silueta real del circuito + podio + parrilla con banda
"dónde acaba si termina") · Sandbox (temp × lluvia, 25 escenarios precalculados) · Honestidad
(backtest 2025, calibración de intervalos, ablación, track record) · **Nerd Data** (71 features/piloto, 9 grupos).

**Honestidad muestra ya "modelo vs parrilla"** (v9): bloque "¿Y si no hubiera modelo?" con barras modelo/parrilla
y modelo/ritmo reciente sobre los mismos finalizadores de 2025, columna "ancla sola" en la ablación, cobertura y
coste del clima leídos de los datos (nada fijo en el HTML). Feedback general de la v6 aún pendiente del usuario.
Últimas críticas atendidas: "P1–P18 es demasiado vago" → rango condicionado a terminar; "la interfaz está fea" → rediseño.

**Idea aparcada**: palanca de parrilla en el sandbox ("¿y si VER sale 10º?") — es lo que de verdad
mueve las probabilidades (el clima apenas: 0.06 puestos). Natural tras la clasificación.

---

## 6. Decisiones y lecciones que no hay que redescubrir

- **Limpieza del 13/09 (refactor)**: se eliminaron la app Flask+PyQt original (`python-app/`, `ml/api/`),
  Streamlit (`ml/dashboard/app.py`) y los scripts de un solo uso (`circuit_trace`, `reparse_era5`,
  `experiment_anchor`). El escaparate es solo el dashboard estático. `FEATURES_PATH` y `CATEGORICAL_COLS`
  viven en `ml/features/build_features.py`; `get_db()` exige `MONGODB_DB` en `.env` (sin default: el default
  a la base vieja fue el incidente del 11/09); el jar Java también apunta por defecto a `-Prod`.

- **No dibujar circuitos por telemetría.** El usuario aportó los SVG oficiales; la herramienta que
  trazaba circuitos desde GPS (commit 224a136) funcionaba pero se eliminó en la limpieza del 13/09. Lección guardada en memoria: comprobar si existe una fuente simple
  antes de construir algo elaborado.
- **Clima aporta poco** al modelo (0.06 puestos). Es un resultado honesto, documentado en el dashboard,
  no un fallo a arreglar. Sensores de pista mejoraron algo; interacciones piloto×clima ya están.
- **La vaguedad era de presentación**: mezclar abandonos en el intervalo lo destruía. Separar
  "si termina" de "probabilidad de no terminar" lo arregló sin tocar el modelo.
- **Ruido residual = 0.5 × error medido** (no 1.0): reordenar ya duplica la dispersión. Medido, no supuesto.
- **Novatos/circuitos nuevos** → NaN en categóricas, el modelo usa forma reciente. Producción conoce
  a todos menos LIN (debutante puro 2026). Madrid es circuito nuevo: sin historial, honestamente vacío.
- **Jolpica se retrasa horas** tras cada sesión. Backoff exponencial ya puesto en ambos clientes.
- **Nombres de circuito**: OpenF1 usa alias distintos ("Catalunya", "Spa-Francorchamps"…). Resuelto
  con `openf1_names` en `circuits.json`; todo join pasa por `_openf1_join.py`. Fue un bug silencioso del 30% de filas.

---

## 7. Siguientes pasos, en orden

1. Recoger feedback general del usuario sobre el dashboard (v9) y ajustar.
2. Fase C (del plan original): automatización tras carrera, README del repo, Makefile, GitHub Pages.
3. Opcionales con valor: palanca de parrilla en sandbox (ahora natural: el modelo ya predice deltas frente a ella);
   ingesta de libres (FP2 long runs — la única fuente de información nueva pre-quali). Descartado con datos:
   red neuronal (no es un problema de capacidad); `race_control` para DNF (los abandonos de 2025 son DSQ y
   averías desde delante, impredecibles a nivel individual; el dado del simulador ya los trata).
