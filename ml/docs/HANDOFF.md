# HANDOFF — estado del proyecto y cómo retomarlo

Última actualización: **2026-09-12, ~17:20 UTC** (sábado del GP de Madrid).
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
| API/Dashboard viejo | Flask blueprint `/ml/*` + Streamlit (diagnóstico interno). Siguen funcionando. |

---

## 3. Trabajo SIN COMMITEAR (importante — commitear al retomar)

```
 M ml/simulation/monte_carlo.py          ← calibración RESIDUAL_SCALE=0.5 + rangos "si termina" + fix identidad pilotos
?? ml/training/calibrate_simulation.py   ← verifica que el intervalo P10-P90 cubre 80% real (antes 88%, inflado)
?? ml/tests/test_simulator_identity.py   ← 3 tests: driver_code no se pierde al anular categorías no vistas
?? ml/dashboard/export_data.py           ← exporta data.json para el dashboard estático
?? ml/dashboard/nerd_data.py             ← 71 features etiquetadas en 9 grupos para la pestaña Nerd Data
?? ml/dashboard/circuit_art.py           ← mapea circuito → SVG oficial F1.com, recolorea a currentColor
?? ml/dashboard/web/                     ← index.html + data.json (el dashboard estático nuevo)
?? ml/ingestion/circuit_trace.py         ← APARCADO: extrae trazado real por GPS. Funciona, pero no es el camino (ver §6)
?? svg/                                  ← 25 SVG de circuitos aportados por el usuario (paquete F1.com)
```

Sugerencia de commits (por capa, estilo del repo: inglés imperativo, cuerpo explicando el *por qué*):
1. Simulator calibration + finisher-conditional ranges + identity fix (+ test + calibrate script)
2. Static dashboard (export_data, nerd_data, circuit_art, web/, svg/)
3. circuit_trace.py — decidir si commitear como "shelved tool" o borrar.

Tests: 19/19 en verde (`./venv/Scripts/python.exe -m pytest ml/tests/ -q`).

---

## 4. EL EXPERIMENTO PENDIENTE (con hora límite: carrera domingo 13/09 13:00 UTC)

Diseñado ayer, a medias ejecutado hoy. Compara tres predicciones de Madrid contra la realidad:

| Momento | Estado |
|---|---|
| **Pre-clasificación** | ✅ Registrada 12/09 09:33 UTC (régimen `pre_quali`, modelo `2026-09-12_production`, clima pronóstico). VER 22.5% victoria. |
| **Post-clasificación** | ⏳ **PENDIENTE.** Quali fue 14:00-15:00 UTC. A las 17:03 UTC Jolpica aún no la publicaba. |
| **Resultado real** | ⏳ Domingo tras la carrera. |

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

---

## 5. Dashboard estático — estado y dirección

**Fichero**: `ml/dashboard/web/index.html` (autocontenido) + `data.json` (305 KB, generado por `export_data.py`).
**Publicado como artefacto** (v6): https://claude.ai/code/artifact/fc54c822-cfad-4491-947e-3e44d2243699
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

**Pendiente de feedback del usuario** sobre la v6 (acaba de publicarse). Últimas críticas atendidas:
"P1–P18 es demasiado vago" → rango condicionado a terminar (P1–P5 + DNF aparte); "la interfaz está fea" → rediseño completo.

**Idea aparcada**: palanca de parrilla en el sandbox ("¿y si VER sale 10º?") — es lo que de verdad
mueve las probabilidades (el clima apenas: 0.06 puestos). Natural tras la clasificación.

---

## 6. Decisiones y lecciones que no hay que redescubrir

- **No dibujar circuitos por telemetría.** El usuario aportó los SVG oficiales; `circuit_trace.py`
  funciona pero quedó aparcado. Lección guardada en memoria: comprobar si existe una fuente simple
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

1. **Commitear** lo de §3.
2. **Ejecutar el experimento** de §4 en cuanto Jolpica publique la clasificación.
3. Domingo: cerrar el experimento con el resultado real.
4. Recoger feedback del usuario sobre el dashboard v6 y ajustar.
5. Fase C (del plan original): automatización tras carrera, README del repo, Makefile, GitHub Pages.
6. Opcionales con valor: palanca de parrilla en sandbox; ingesta de libres (FP2 long runs); `race_control` para safety cars (DNF AUC 0.53 es el punto débil).
