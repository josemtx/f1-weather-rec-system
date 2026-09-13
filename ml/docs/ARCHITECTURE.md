# Arquitectura del sistema predictivo F1

## Objetivo

Sistema de predicción de resultados de Fórmula 1 con ML riguroso (XGBoost),
simulación Monte Carlo y clima dual-fuente, construido sobre la ingesta
ya existente (Java → MongoDB) de OpenF1 y OpenWeatherMap.

## Decisión de alcance: modelo "era moderna" (2023-2026)

**Decisión (2026-09-11)**: el modelo de producción se entrena únicamente con
carreras de **2023 en adelante**, no con el histórico completo 2018-2026.

**Por qué**: la riqueza real de datos no cambia en 2021 (como se asumió al
principio) sino en **2023**, cuando empieza la cobertura de OpenF1:

| Periodo | Jolpica (resultado, grid, quali) | Sprints | OpenF1 (laps/stints/pit) |
|---|---|---|---|
| 2018-2020 | ✅ | ❌ (no existían) | ❌ |
| 2021-2022 | ✅ | ✅ (parcial) | ❌ |
| **2023-2026** | ✅ | ✅ | **✅** |

Entrenar sobre 2018-2026 mezclando eras dejaba el 55-57% de las filas con
NaN en las features más informativas (estrategia de neumáticos, ritmo),
además de mezclar dos reglamentos técnicos distintos (2022 introdujo los
coches de efecto suelo — Mercedes dominante hasta 2021, Red Bull dominante
desde 2022).

**2018-2022 se mantiene en MongoDB** (no se borra) y se usa como
"calentamiento" de las medias móviles: las features de forma reciente de
las primeras carreras de 2023 miran hacia atrás a 2022 en vez de arrancar
en frío, pero esas filas de 2018-2022 nunca son objetivo de entrenamiento.

## Cobertura real de OpenF1 (verificado, no asumido)

Verificado con `curl` directo el 2026-09-11: `sessions?year=2018..2022`
devuelve `{"detail":"No results found."}` para los 5 años — cobertura
real desde 2023 únicamente. Por eso `grid_position` y el resultado final
vienen de **Jolpica-F1** (sucesor de Ergast) para TODOS los años, no de
OpenF1 (que no los tiene fuera de 2023+).

OpenF1 expone **4 sesiones por fin de semana**, no solo "Race":

```
Qualifying | Sprint Qualifying | Sprint | Race
```

La ingesta original solo traía `Race`. Se extendió (`AppFormula1.java`) a
traer las 4 (2023+ únicamente) para tener vueltas/stints/paradas reales de
clasificación y sprint, no solo el resultado final.

**IMPORTANTE para cualquier código nuevo que lea `laps`/`stints`/`pit`/`positions`/`weather`**:
la colección `sessions` ahora mezcla los 4 tipos. Todo join debe filtrar
por `session_name` explícitamente (ver `ml/features/feature_defs/_openf1_join.py`,
que por defecto restringe a `"Race"` — cualquier consumidor que necesite
Qualifying/Sprint debe pasarlo explícitamente).

## Bugs encontrados y corregidos esta sesión

1. **`getSessions()` sin try-catch por año** (Formula1Service.java): un
   único año sin datos abortaba toda la ingesta. Corregido antes de
   verificar que 2018-2022 realmente no tienen datos.
2. **`MONGO_DB` no heredado en shell de larga duración**: una sesión de
   terminal abierta antes de fijar la variable de entorno a nivel de
   Usuario en Windows no la recoge — dos ingestas Java escribieron en la
   base vieja `F1-WeatherRec` por error. Mitigado fijando `MONGO_DB`
   explícitamente en cada comando, y en `run-weather-ingestion.ps1`.
3. **ERA5 (Copernicus) devuelve un `.zip`** aunque se pida `data_format=netcdf`
   directamente — el `.nc` real está dentro. `_resolve_actual_netcdf()` en
   `ingest_era5.py` lo detecta y extrae de forma transparente.
4. **`finished` mal calculado en Jolpica**: se marcaba `"Lapped"` (terminó
   la carrera, una vuelta por detrás del líder) como abandono, inflando la
   tasa de DNF de test (31%) muy por encima de la de train (18.8%) y
   degradando el clasificador de DNF. Corregido usando `positionText.isdigit()`
   (numérico = clasificado, letra = no clasificado) en vez de parsear el
   texto libre de `status`.
5. **Sin `random_state` en XGBoost**: cada reentrenamiento daba métricas
   ligeramente distintas sin que fuera señal real. Corregido y verificado
   bit a bit (dos entrenamientos consecutivos, mismos números exactos).
6. **Rate limiting sostenido en corridas largas**: el backoff lineal
   (1,2,3,4,5s, 5 intentos) resultaba insuficiente pasada cierta duración,
   dejando huecos reales de datos en 2024-2026 (justo los años del modelo
   B). Corregido con backoff exponencial (techo 20s) y más intentos (8) en
   ambos clientes (Jolpica y OpenF1).

## Dos modelos con roles distintos (decision 2026-09-12)

`ml/models/registry.json` apunta a dos versiones que cumplen funciones
incompatibles entre si:

- **`evaluation`**: entrenado con 2023-2024, dejando 2025 fuera. Sus metricas
  son creibles precisamente porque nunca vio ese anio. Es de donde salen
  todos los numeros que se reportan.
- **`production`**: misma receta entrenada con TODO (2023-2025). Predice
  mejor porque conoce el grid mas reciente, pero no se le pueden medir notas
  sobre datos que ya ha visto. Su calidad estimada es la del modelo de
  evaluacion. Hereda de el los coeficientes de calibracion.

Efecto medible: de los pilotos del grid 2026, el modelo de evaluacion
desconoce ANT, BOR, LIN y HAD; el de produccion solo LIN (debutante puro de
2026). Una categoria desconocida no rompe la prediccion -- se mapea a NaN y
el modelo se apoya en la forma reciente del piloto en vez de en su identidad.

## Prediccion de carreras futuras

`ml/features/upcoming.py` genera una fila "fantasma" por piloto (circuito,
fecha y parrilla conocidos; resultado vacio) y la anade al final del
historico. A partir de ahi el pipeline de features corre sin cambios: el
`shift(1)` del anti-fuga hace que esa fila vea todo el pasado y nada de si
misma. No hay una segunda implementacion de features que pueda divergir.

De ese diseno salen solos dos regimenes de prediccion, medidos sobre 2025
con `ml/training/ablation.py`:

| Momento | MAE | Aciertos de podio |
|---|---|---|
| Sabado noche (con clasificacion) | 3.34 | 2.04 / 3 |
| Viernes (sin clasificacion) | 3.83 | 1.54 / 3 |

Es decir: conocer la parrilla vale ~0.5 puestos de precision.

## Fuentes de clima (dual, ver plan original)

- **NASA POWER**: primaria, sin key, cubre 2018-2026 completo por circuito
  (no solo días de carrera — el año calendario entero, lo que permite
  cubrir también sábados de clasificación y sprint sin ingesta adicional).
- **Copernicus ERA5-Land**: secundaria, más precisa, requiere cuenta CDS.
  Solo se pidieron fechas de carrera (para minimizar cola) — clasificación
  y sprint usan NASA POWER.
- **Coalescencia**: `climate_historical` guarda ambas fuentes como filas
  independientes (`source` es parte de la clave única); ERA5 gana sobre
  NASA POWER cuando ambas existen para el mismo circuito+fecha, nunca se
  sobreescriben entre sí.

## Limitaciones documentadas (no resueltas, conocidas)

- **`driver_quali_gap_to_pole_pct`**: viable ahora que hay datos de
  clasificación (Jolpica `/qualifying.json`), implementado en `qualifying.py`.
- **Safety car / VSC**: no hay ingesta de `race_control` (OpenF1). Se usa
  un proxy por tipo de circuito (`circuit_avg_safety_car_proxy`) en vez de
  datos reales de incidentes.
- **Predicción de carreras futuras**: el pipeline actual solo construye
  features para carreras que YA ocurrieron (usa `climate_historical`,
  clima real, no pronóstico). Conectar `forecast_data` (OpenWeatherMap,
  ya se ingiere a diario) para predecir una carrera de mañana es trabajo
  pendiente, igual que decidir qué hacer con `grid_position` antes de que
  exista (predicción previa a clasificación).
- **Interacción explícita piloto×clima**: implementada como categoría J
  (`weather_interaction.py`: media en mojado/seco/calor y sus deltas, por
  piloto y equipo). Aporta ~0.06 puestos: con ~5% de carreras mojadas hay
  3-4 en el train, y es un techo de datos, no de diseño.
- **El regresor no supera a la parrilla si predice posicion absoluta**
  (medido el 2026-09-12 con `residual_diagnosis.py`: 2024 3.47 vs 2.91,
  2025 3.36 vs 3.34). Por eso predice deltas frente a un ancla, solo con
  finalizadores y con objetivo MAE (`ml/training/anchor.py`, `train.py`).
  Pre-clasificacion sigue valiendo lo que su ancla (ritmo reciente); la
  ganancia real esta en el regimen post-clasificacion (-0.4 puestos en 2025).
