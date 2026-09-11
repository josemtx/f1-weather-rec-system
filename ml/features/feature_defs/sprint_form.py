"""Categoria I: forma en sprints (~3 features, solo fines de semana con sprint,
2021+). El rolling se calcula sobre la SECUENCIA DE SPRINTS de cada piloto
(no sobre todas las carreras) -- "ultimos 5 sprints", no "ultimas 5 rondas".
NaN en fines de semana sin sprint, que es la mayoria del calendario.
"""

import pandas as pd

from ml.features.anti_leakage import rolling_mean_shifted, rolling_rate_shifted


def _weekend_sprint_context(db, context_df: pd.DataFrame) -> pd.DataFrame:
    """Dos datos del fin de semana en curso, ambos conocidos ANTES de la carrera:

      weekend_has_sprint          -> sale del calendario, asi que existe
                                     tambien para carreras futuras. Un fin de
                                     semana con sprint tiene menos libres y
                                     distinta gestion de neumaticos.
      driver_sprint_position_this_weekend -> el sprint se corre el sabado,
                                     antes del domingo: su resultado es
                                     informacion legitima para predecir la
                                     carrera, igual que la clasificacion.
    """
    schedule = list(db["race_schedule"].find({}, {"_id": 0, "year": 1, "round": 1, "sprint_date": 1}))
    out = pd.DataFrame(index=context_df.index)

    if schedule:
        sched = pd.DataFrame(schedule)
        sched["weekend_has_sprint"] = sched["sprint_date"].notna().astype(float)
        merged = context_df[["year", "round"]].merge(
            sched[["year", "round", "weekend_has_sprint"]], on=["year", "round"], how="left"
        )
        out["weekend_has_sprint"] = merged["weekend_has_sprint"].fillna(0.0).to_numpy()
    else:
        out["weekend_has_sprint"] = 0.0

    results = list(db["sprint_results"].find({}, {"_id": 0, "year": 1, "round": 1, "driver_code": 1, "sprint_position": 1}))
    if results:
        sprint_res = pd.DataFrame(results)
        merged = context_df[["year", "round", "driver_code"]].merge(
            sprint_res, on=["year", "round", "driver_code"], how="left"
        )
        out["driver_sprint_position_this_weekend"] = merged["sprint_position"].to_numpy()
    else:
        out["driver_sprint_position_this_weekend"] = float("nan")

    return out


def build_sprint_form(db, context_df: pd.DataFrame) -> pd.DataFrame:
    weekend = _weekend_sprint_context(db, context_df)

    rows = list(db["sprint_results"].find({}, {"_id": 0}))
    if not rows:
        cols = ["driver_avg_sprint_finish_last5", "driver_sprint_podium_rate_last5", "team_avg_sprint_finish_last5"]
        empty = pd.DataFrame(index=context_df.index, columns=cols)
        return pd.concat([empty, weekend], axis=1)

    sprint = pd.DataFrame(rows)
    sprint["sprint_date"] = pd.to_datetime(sprint["sprint_date"])
    sprint["_podium"] = sprint["sprint_position"].fillna(99) <= 3

    sprint["driver_avg_sprint_finish_last5"] = rolling_mean_shifted(
        sprint, "driver_code", "sprint_date", "sprint_position", 5
    )
    sprint["driver_sprint_podium_rate_last5"] = rolling_rate_shifted(
        sprint, "driver_code", "sprint_date", "_podium", 5
    )
    sprint["team_avg_sprint_finish_last5"] = rolling_mean_shifted(
        sprint, "constructor_id", "sprint_date", "sprint_position", 5
    )

    cols = ["year", "round", "driver_code", "driver_avg_sprint_finish_last5",
            "driver_sprint_podium_rate_last5", "team_avg_sprint_finish_last5"]
    merged = context_df.merge(sprint[cols], on=["year", "round", "driver_code"], how="left")

    rolling = merged[[
        "driver_avg_sprint_finish_last5", "driver_sprint_podium_rate_last5", "team_avg_sprint_finish_last5"
    ]].set_axis(context_df.index)

    return pd.concat([rolling, weekend], axis=1)
