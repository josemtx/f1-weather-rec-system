"""Helpers compartidos para features de 'forma historica' sin fuga temporal.

Regla vinculante (ver plan): para la carrera R en fecha d_R, toda feature de
forma historica de un piloto/equipo debe usar solo filas con fecha < d_R.
Se implementa estructuralmente con shift(1) antes de rolling/expanding, no
con un filtro `WHERE date < d_R` reimplementado por feature -- shift(1)
excluye mecanicamente la fila actual sin importar como se llame a la funcion.

Todas las funciones asumen que `df` tiene una fila por (group, fecha de
carrera) y que NO hay carreras duplicadas por grupo+fecha (si las hay,
agregar antes de llamar). El resultado se alinea al indice original de `df`.
"""

import pandas as pd


def _sorted(df: pd.DataFrame, group_col: str, date_col: str) -> pd.DataFrame:
    return df.sort_values([group_col, date_col], kind="mergesort")


def _numeric(series: pd.Series) -> pd.Series:
    """Fuerza float64 antes de hacer rolling.

    Las filas de carreras futuras (ml/features/upcoming.py) traen el
    resultado vacio; si llega como pd.NA/None la columna pasa a dtype object
    y pandas no puede hacer rolling sobre ella. Convertir aqui, en el unico
    punto por el que pasan todas las features historicas, evita tener que
    recordarlo en cada modulo de feature_defs.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64")


def rolling_mean_shifted(
    df: pd.DataFrame, group_col: str, date_col: str, value_col: str, window: int
) -> pd.Series:
    """Media movil de las ultimas `window` carreras ANTERIORES a cada fila."""
    ordered = _sorted(df, group_col, date_col)
    shifted = _numeric(ordered[value_col]).groupby(ordered[group_col]).shift(1)
    result = shifted.groupby(ordered[group_col]).rolling(window, min_periods=1).mean()
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)


def rolling_rate_shifted(
    df: pd.DataFrame, group_col: str, date_col: str, bool_col: str, window: int
) -> pd.Series:
    """Tasa (0-1) de un evento booleano en las ultimas `window` carreras anteriores."""
    return rolling_mean_shifted(df, group_col, date_col, bool_col, window)


def rolling_sum_shifted(
    df: pd.DataFrame, group_col: str, date_col: str, value_col: str, window: int
) -> pd.Series:
    """Suma de las ultimas `window` carreras ANTERIORES a cada fila (p.ej. puntos)."""
    ordered = _sorted(df, group_col, date_col)
    shifted = _numeric(ordered[value_col]).groupby(ordered[group_col]).shift(1)
    result = shifted.groupby(ordered[group_col]).rolling(window, min_periods=1).sum()
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)


def expanding_mean_shifted(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> pd.Series:
    """Media acumulada de TODAS las carreras anteriores (career-to-date / circuit history)."""
    ordered = _sorted(df, group_col, date_col)
    shifted = _numeric(ordered[value_col]).groupby(ordered[group_col]).shift(1)
    result = shifted.groupby(ordered[group_col]).expanding().mean()
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)


def expanding_sum_shifted(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> pd.Series:
    """Suma acumulada de TODAS las carreras anteriores (p.ej. victorias en carrera)."""
    ordered = _sorted(df, group_col, date_col)
    shifted = _numeric(ordered[value_col]).groupby(ordered[group_col]).shift(1)
    result = shifted.groupby(ordered[group_col]).expanding().sum()
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)


def conditional_expanding_mean_shifted(
    df: pd.DataFrame, group_col: str, date_col: str, value_col: str, condition_col: str
) -> pd.Series:
    """Media de `value_col` sobre las carreras ANTERIORES que cumplian una
    condicion (p.ej. "posicion media de este piloto en carreras con lluvia").

    Truco: se anulan (NaN) los valores que no cumplen la condicion y se
    reutiliza el expanding de siempre, que ignora NaN. Asi la regla anti-fuga
    sigue siendo exactamente la misma -- un unico shift(1) -- en vez de una
    segunda implementacion que pudiera divergir.
    """
    masked = _numeric(df[value_col]).where(df[condition_col].astype("boolean").fillna(False))
    tmp = df.assign(_masked=masked)
    return expanding_mean_shifted(tmp, group_col, date_col, "_masked")


def conditional_expanding_count_shifted(
    df: pd.DataFrame, group_col: str, date_col: str, condition_col: str
) -> pd.Series:
    """Cuantas carreras anteriores del grupo cumplian la condicion: mide
    cuanta evidencia hay detras de la media condicional."""
    flags = df[condition_col].astype("boolean").fillna(False).astype("float64")
    tmp = df.assign(_flag=flags)
    return expanding_sum_shifted(tmp, group_col, date_col, "_flag")


def expanding_count_shifted(df: pd.DataFrame, group_col: str, date_col: str) -> pd.Series:
    """Numero de carreras previas del grupo (para saber si hay suficiente historial)."""
    ordered = _sorted(df, group_col, date_col)
    counts = ordered.groupby(group_col).cumcount()
    return counts.reindex(df.index)
