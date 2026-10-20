"""Construcción de los entornos de entrenamiento, validación y held-out.

Este módulo une los datos y las variables de mercado con el entorno de
simulación. El proceso se realiza en los siguientes pasos:

1. Se carga el dataset interpolado y se divide con las fechas de corte definidas
   en configs/project.toml.
2. Se calculan los indicadores con FeatureEngineer. Dado que todas las ventanas
   miran únicamente hacia atrás, los indicadores se pueden calcular sobre la
   serie completa y luego recortar por partición sin introducir información del
   futuro. Las primeras filas de validación y de held-out utilizan precios de la
   partición anterior, lo cual es válido porque corresponden a datos pasados.
   Esta propiedad puede comprobarse modificando los precios y el volumen
   posteriores a una vela k, recalculando los indicadores y verificando que las
   filas hasta k no cambian.
3. Se descartan las filas del periodo de calentamiento. La media móvil de 168
   horas necesita una semana de historia, por lo que las primeras filas de
   entrenamiento no tienen todos los indicadores disponibles.
4. El escalador se ajusta solo con las filas de entrenamiento y después se aplica
   a las tres particiones. Como consecuencia, las variables quedan con media 0 y
   desviación 1 en entrenamiento, pero no en validación ni en held-out, lo cual
   es el comportamiento esperado.
5. Se entrega un TradingEnvironment por cada partición.

La posición del agente no pasa por el escalador. Esta variable no es una
propiedad del mercado sino del propio agente, por lo que el entorno la agrega en
su escala natural (0, 0.5 o 1). Por la misma razón, portfolio_value no se usa
como variable de observación.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DEFAULT_CONFIG_PATH, load_ohlcv, load_split_config, split_by_dates
from src.environment import TradingEnvironment
from src.features import FeatureEngineer

# Variables que describen el estado del agente y no el mercado.
AGENT_STATE_COLUMNS = ("position", "portfolio_value")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarketData:
    """Datos de una partición, listos para crear un entorno."""

    prices: np.ndarray
    features: np.ndarray
    timestamps: list
    feature_names: list[str]


def build_market_data(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[dict[str, MarketData], FeatureEngineer]:
    """Devuelve los datos de cada partición y el FeatureEngineer ya ajustado."""
    with Path(config_path).open("rb") as config_file:
        config = tomllib.load(config_file)

    data_path = PROJECT_ROOT / config["preprocessing"]["interpolated_path"]
    frame, _ = load_ohlcv(data_path, config["dataset"]["expected_frequency"])
    frame = frame.reset_index(drop=True)

    engineer = FeatureEngineer()
    indicators = engineer.compute_indicators(frame)
    market_columns = [
        column
        for column in engineer.feature_columns
        if column not in AGENT_STATE_COLUMNS
    ]
    indicators = indicators[market_columns]

    partitions = split_by_dates(frame, load_split_config(config_path))

    # Se eliminan las filas del periodo de calentamiento, es decir, aquellas en
    # las que los indicadores aún no tienen suficiente historia y resultan NaN.
    valid = indicators.notna().all(axis=1)
    clean_rows = {name: part.index[valid.loc[part.index]] for name, part in partitions.items()}

    # El escalador se ajusta únicamente con las filas de entrenamiento, para que
    # validación y held-out no influyan en la normalización.
    engineer.fit_scaler(indicators.loc[clean_rows["train"]])

    result: dict[str, MarketData] = {}
    for name, rows in clean_rows.items():
        scaled = engineer.transform(indicators.loc[rows])
        result[name] = MarketData(
            prices=frame.loc[rows, "close"].to_numpy(dtype=np.float64),
            features=scaled.to_numpy(dtype=np.float64),
            timestamps=list(frame.loc[rows, "timestamp"]),
            feature_names=market_columns,
        )
    return result, engineer


def make_environments(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    **env_kwargs,
) -> dict[str, TradingEnvironment]:
    """Crea un entorno por partición con los mismos costos y el mismo capital.

    Los argumentos adicionales (reward_fn, transaction_cost_bps, entre otros) se
    pasan a todos los entornos. Si se desea entrenar con episodios cortos y
    arranque al azar, se puede crear por separado un entorno de train con
    episode_length y random_start.
    """
    data, _ = build_market_data(config_path)
    return {
        name: TradingEnvironment(
            prices=part.prices,
            features=part.features,
            timestamps=part.timestamps,
            **env_kwargs,
        )
        for name, part in data.items()
    }
