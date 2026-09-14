# F1 race prediction — with an honest track record

[![tests](https://github.com/josemtx/f1-weather-rec-system/actions/workflows/ci.yml/badge.svg)](https://github.com/josemtx/f1-weather-rec-system/actions/workflows/ci.yml)
[![dashboard](https://github.com/josemtx/f1-weather-rec-system/actions/workflows/pages.yml/badge.svg)](https://josemtx.github.io/f1-weather-rec-system/)

Probabilistic Formula 1 race predictions — win, podium and points chances per
driver, where they finish *if they finish*, and how likely they are not to —
served as a static dashboard that shows how well the model does against the
simplest possible alternative.

**Live dashboard → https://josemtx.github.io/f1-weather-rec-system/**

[![Dashboard: next race, most likely podium, full grid with finishing bands](.github/readme/dashboard.jpg)](https://josemtx.github.io/f1-weather-rec-system/)

## Results, measured out of sample

Every number below comes from the full 2025 season, predicted by a model that
never saw it during training.

| | Mean error (positions) | Copying the anchor instead |
|---|---|---|
| After qualifying (anchor: the grid) | **2.49** | 2.88 |
| Before qualifying (anchor: recent pace) | 3.11 | 3.07 |

- Once the grid is known, the model beats "predict = grid" by ~0.4 positions
  and gets 2.3 of the 3 podium finishers right per race.
- Before qualifying it knows nothing beyond a driver's recent pace, and the
  dashboard says so.
- The 80 % intervals cover **80.4 %** of real results — calibrated, not assumed.
- Predictions are written to the database *before* each race, with the
  information available at that moment, and cannot be rewritten afterwards.

[![Honesty tab: model vs grid over the same 2025 finishers](.github/readme/honesty.jpg)](https://josemtx.github.io/f1-weather-rec-system/)

## How it is built

```
Java · OpenF1 + OpenWeatherMap  ─┐
Python · Jolpica/Ergast          ├─►  MongoDB  ─►  81 leak-free features  ─►  XGBoost
Python · NASA POWER + ERA5      ─┘                 (10 groups, shift(1))       (4 targets)
                                                                                   │
        static dashboard  ◄─  data.json  ◄─  Monte Carlo simulator  ◄──────────────┘
        (GitHub Pages)                       weather · reliability · pit stops · calibrated noise
```

| Layer | What it does |
|---|---|
| `java-app/` | Ingests sessions, laps, stints, pit stops and track sensors from OpenF1; daily weather forecasts per circuit (Windows Task Scheduler) |
| `ml/jolpica/`, `ml/ingestion/` | Results, qualifying and calendar 2018–2026; historical climate (NASA POWER, Copernicus ERA5); circuit catalogue |
| `ml/features/` | 92 columns in 10 groups; every rolling statistic is shifted one race so nothing from the future leaks in; "ghost rows" let future races flow through the same pipeline |
| `ml/training/` | Regressor anchored to the grid (it predicts the delta, not the position), calibrated classifiers, walk-forward backtest, ablation by information regime, interval calibration, residual diagnosis |
| `ml/simulation/` | Monte Carlo over the model: weather, reliability, pit-stop jitter and residual noise sized from the measured error |
| `ml/predictions/` | Auditable prediction log with the information regime (before / after qualifying) |
| `ml/dashboard/` | Exporter to `data.json` and the dependency-free HTML page |

Stack: Python 3.12 (pandas, XGBoost, scikit-learn, SHAP), Java 17 + Maven,
MongoDB, vanilla HTML/CSS/JS. 24 unit tests, no database needed to run them.

## Decisions worth knowing about

- **Predict the delta from an anchor, not the absolute position.** Measured
  first: a regressor predicting finishing position outright did *not* beat
  "predict = grid" (2024: 3.47 vs 2.91). Anchoring flips that.
- **Retirements never enter the regressor.** The simulator rolls them
  separately. Mixing them in biased every finisher by a full position and made
  the intervals meaningless ("P1–P18").
- **Weather adds ~0.02 positions.** With ~5 % of races wet, that is a data
  ceiling, not a design flaw — it is documented on the dashboard rather than
  hidden.
- **Live predictions had a train/serve skew** the backtest could not see: for a
  future race the official grid does not exist yet and arrived as NaN. Found
  through a SHAP breakdown, fixed by filling from qualifying position
  (`ml/training/anchor.py`).

More in [ml/docs/ARCHITECTURE.md](ml/docs/ARCHITECTURE.md) (Spanish).

<details>
<summary><b>Run it yourself</b></summary>

Requirements: Python 3.11+, Java 17 + Maven, a local MongoDB.

```bash
python -m venv venv && ./venv/Scripts/pip install -r ml/requirements.txt
cp .env.example .env            # MONGODB_URI, MONGODB_DB, OPENWEATHER_API_KEY, CDS_URL/CDS_KEY

# Data
(cd java-app && mvn clean package && java -jar target/F1-WeatherRec.jar f1)
python -m ml.ingestion.load_circuits
python -m ml.jolpica.ingest_jolpica
python -m ml.ingestion.ingest_climate            # NASA POWER
python -m ml.ingestion.ingest_era5               # optional, needs a CDS account

# Model
python -m ml.features.build_features
python -m ml.training.train
python -m ml.training.ablation
python -m ml.training.calibrate_simulation       # and `... <version> pre_quali`

# Every race weekend
python -m ml.predictions.predict_race            # Friday, and again after qualifying
python -m ml.dashboard.export_data               # regenerates ml/dashboard/web/data.json
python -m pytest ml/tests -q
```

The dashboard is a self-contained page: serve `ml/dashboard/web/` as is
(GitHub Pages does it from the `dashboard` workflow on every push).

</details>
