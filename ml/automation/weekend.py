"""El fin de semana de carrera sin manos: mira calendario y base de datos y hace lo que falte.

No sabe que dia es. Cada paso es idempotente, asi que se puede lanzar cada
hora (Task Scheduler, ver run-weekend.ps1) y solo actua cuando hay algo nuevo:

  1. Ingesta Jolpica del anio en curso (resultados, clasificacion, sprint).
     Jolpica tarda horas tras cada sesion y a veces esta caido: si falla, se
     sigue con lo que hay y la proxima pasada lo reintenta.
  2. Si acaba de entrar el resultado de una carrera, telemetria OpenF1 de
     esas sesiones (jar Java, modo incremental) para las features de la
     siguiente.
  3. Si la proxima carrera esta a pocos dias y aun no hay prediccion
     registrada para el regimen de informacion actual (pre_quali /
     post_sprint / post_quali, deducido de los datos), se predice y registra.
     Cada regimen se registra UNA vez: es el track record.
  4. Si algo cambio, se regenera data.json; si data.json cambio, commit y
     push. GitHub Pages despliega solo.

Uso: python -m ml.automation.weekend [--dry-run] [--days N]
"""

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.dashboard import export_data
from ml.features.upcoming import get_next_race
from ml.jolpica.ingest_jolpica import ingest as ingest_jolpica
from ml.predictions.predict_race import predict_and_store
from ml.predictions.prediction_log import detect_information_regime

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
JAR = REPO_ROOT / "java-app" / "target" / "F1-WeatherRec.jar"
DATA_JSON = Path("ml/dashboard/web/data.json")
DAYS_AHEAD = 3


def _results_count(db, year: int) -> int:
    return db["historical_results"].count_documents({"year": year})


def step_ingest_jolpica(db, year: int) -> bool:
    """True si entro el resultado de alguna carrera nueva."""
    before = _results_count(db, year)
    try:
        _, failed = ingest_jolpica([year])
    except Exception as e:  # red caida, 429... se reintenta en la proxima pasada
        logger.warning(f"Jolpica no disponible ({e}); se sigue con los datos existentes")
        return False
    if failed:
        logger.warning("Jolpica fallo para el anio en curso; se sigue con los datos existentes")
    new_races = _results_count(db, year) > before
    if new_races:
        logger.info("Ha entrado el resultado de una carrera nueva")
    return new_races


def step_openf1_incremental() -> None:
    if not JAR.exists():
        logger.warning(f"No existe {JAR}; sin telemetria OpenF1 (mvn clean package en java-app)")
        return
    logger.info("Telemetria OpenF1 (incremental)...")
    subprocess.run(["java", "-jar", str(JAR), "f1", "incremental"], cwd=REPO_ROOT, check=False)


def step_predict_if_due(db, today: date, days_ahead: int) -> tuple[bool, bool]:
    """(estamos en semana de carrera, se registro una prediccion nueva)."""
    nxt = get_next_race(db)
    if nxt is None:
        logger.info("No hay carrera futura en el calendario")
        return False, False
    year, round_number = int(nxt["year"]), int(nxt["round"])
    days_to_race = (pd.Timestamp(nxt["race_date"]).date() - today).days
    if days_to_race > days_ahead:
        logger.info(f"Proxima carrera {year} R{round_number} en {days_to_race} dias; aun no toca")
        return False, False

    regime = detect_information_regime(db, year, round_number)
    already = db["predictions"].count_documents({"year": year, "round": round_number, "information_regime": regime})
    if already:
        logger.info(f"{year} R{round_number}: prediccion {regime} ya registrada")
        return True, False

    logger.info(f"{year} R{round_number}: registrando prediccion {regime}")
    predict_and_store(db, year, round_number)
    return True, True


def step_publish(dry_run: bool) -> None:
    export_data.main()
    changed = subprocess.run(["git", "status", "--porcelain", str(DATA_JSON)], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
    if not changed:
        logger.info("data.json sin cambios; nada que publicar")
        return
    if dry_run:
        logger.info("data.json cambio (dry-run: sin commit ni push)")
        return
    subprocess.run(["git", "add", str(DATA_JSON)], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"Dashboard data: automatic update {date.today().isoformat()}"],
                   cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=REPO_ROOT, check=True)
    logger.info("data.json publicado (commit + push); GitHub Pages despliega solo")


def main(dry_run: bool = False, days_ahead: int = DAYS_AHEAD) -> None:
    db = get_db()
    today = date.today()

    new_results = step_ingest_jolpica(db, today.year)
    if new_results:
        step_openf1_incremental()
    race_week, new_prediction = step_predict_if_due(db, today, days_ahead)

    # En semana de carrera se regenera siempre: el pronostico entra a diario
    # y la portada simula en vivo con el ultimo. Fuera de ella, solo si hay
    # resultados nuevos que cierren el track record.
    if new_results or new_prediction or race_week:
        step_publish(dry_run)
    else:
        logger.info("Sin novedades")


if __name__ == "__main__":
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="no hace commit ni push")
    parser.add_argument("--days", type=int, default=DAYS_AHEAD, help="dias antes de la carrera a partir de los que se predice")
    args = parser.parse_args()
    try:
        main(dry_run=args.dry_run, days_ahead=args.days)
    except Exception:
        logger.exception("weekend fallo")
        sys.exit(1)
