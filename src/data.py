"""OHLCV loading, data-quality checks, and chronological dataset splitting."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "project.toml"


@dataclass(frozen=True)
class DataQualityReport:
    """Quality findings measured before cleaning, plus the resulting row count."""

    source_rows: int
    clean_rows: int
    start: str
    end: str
    input_was_chronological: bool
    duplicate_rows: int
    rows_with_nulls: int
    null_values_by_column: dict[str, int]
    missing_timestamps: int
    missing_timestamp_examples: list[str]
    rows_by_series: dict[str, int]
    missing_timestamps_by_series: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        raise ValueError(
            f"Formato no soportado: {suffix}. Use CSV o pickle confiable (.pkl/.pickle)."
        )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"El dataset debe contener un DataFrame, no {type(frame).__name__}.")
    return frame


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    normalized = {column: str(column).strip().lower() for column in result.columns}
    result = result.rename(columns=normalized)
    aliases = {"date": "timestamp", "datetime": "timestamp"}
    for old, new in aliases.items():
        if old in result.columns and new not in result.columns:
            result = result.rename(columns={old: new})
    if "timestamp" not in result.columns:
        if isinstance(result.index, pd.DatetimeIndex):
            result.index.name = "timestamp"
            result = result.reset_index()
        else:
            raise ValueError(
                "No se encontró la columna timestamp ni un índice DatetimeIndex."
            )

    missing = sorted(set(REQUIRED_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}. Se requieren {list(REQUIRED_COLUMNS)}."
        )
    columns = [*REQUIRED_COLUMNS]
    columns.extend(
        column
        for column in ("symbol", "exchange", "is_interpolated")
        if column in result.columns
    )
    return result.loc[:, columns]


def load_ohlcv(
    path: str | Path,
    expected_frequency: str = "1h",
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load OHLCV, report quality issues, and apply deterministic basic cleaning.

    Duplicate timestamps keep their last input row. Rows with any null required
    field are removed. Missing candles are reported but never interpolated.
    Invalid timestamps, non-finite/negative values, and inconsistent OHLC bars
    raise errors rather than being silently repaired.
    """
    dataset_path = Path(path)
    frame = _normalize_columns(_read_frame(dataset_path))
    source_rows = len(frame)

    parsed_timestamp = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    invalid_timestamps = int(parsed_timestamp.isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Hay {invalid_timestamps} timestamps inválidos.")
    frame["timestamp"] = parsed_timestamp

    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    null_values = frame.loc[:, REQUIRED_COLUMNS].isna().sum()
    rows_with_nulls = int(frame.loc[:, REQUIRED_COLUMNS].isna().any(axis=1).sum())
    series_columns = [
        column for column in ("exchange", "symbol") if column in frame.columns
    ]
    for column in series_columns:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"La columna {column} contiene valores vacíos.")

    duplicate_key = [*series_columns, "timestamp"]
    chronological_columns = ["timestamp", *series_columns]
    chronological = bool(
        pd.MultiIndex.from_frame(frame[chronological_columns]).is_monotonic_increasing
    )
    duplicate_rows = int(frame.duplicated(subset=duplicate_key, keep=False).sum())

    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    frame = frame.drop_duplicates(subset=duplicate_key, keep="last")
    sort_columns = ["timestamp", *series_columns]
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("El dataset no contiene filas válidas después de eliminar nulos.")

    values = frame.loc[:, REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV contiene valores no finitos.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Los precios OHLC deben ser positivos.")
    if (frame["volume"] < 0).any():
        raise ValueError("El volumen no puede ser negativo.")
    invalid_bars = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if invalid_bars.any():
        raise ValueError(f"Se encontraron {int(invalid_bars.sum())} velas OHLC inconsistentes.")

    groups = (
        frame.groupby(series_columns, sort=True, dropna=False)
        if series_columns
        else [(None, frame)]
    )
    missing_examples_by_series: dict[str, list[str]] = {}
    missing_timestamps_by_series: dict[str, int] = {}
    for key, series_frame in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        label = " / ".join(
            str(value) for value in key_values if value is not None
        ) or "dataset"
        expected_timestamps = pd.date_range(
            start=series_frame["timestamp"].iloc[0],
            end=series_frame["timestamp"].iloc[-1],
            freq=expected_frequency,
        )
        observed_timestamps = pd.DatetimeIndex(series_frame["timestamp"])
        missing_values = expected_timestamps.difference(observed_timestamps)
        missing_timestamps_by_series[label] = len(missing_values)
        missing_examples_by_series[label] = [
            timestamp.isoformat() for timestamp in missing_values[:10]
        ]
    missing_timestamps = sum(missing_timestamps_by_series.values())
    missing_examples = [
        f"{series}: {timestamp}"
        for series, examples in missing_examples_by_series.items()
        for timestamp in examples
    ][:10]
    rows_by_series: dict[str, int] = {}
    for key, count in (
        frame.groupby(series_columns, dropna=False).size().items()
        if series_columns
        else [(None, len(frame))]
    ):
        key_values = key if isinstance(key, tuple) else (key,)
        label = " / ".join(
            str(value) for value in key_values if value is not None
        ) or "dataset"
        rows_by_series[label] = int(count)

    report = DataQualityReport(
        source_rows=source_rows,
        clean_rows=len(frame),
        start=frame["timestamp"].iloc[0].isoformat(),
        end=frame["timestamp"].iloc[-1].isoformat(),
        input_was_chronological=chronological,
        duplicate_rows=duplicate_rows,
        rows_with_nulls=rows_with_nulls,
        null_values_by_column={key: int(value) for key, value in null_values.items()},
        missing_timestamps=missing_timestamps,
        missing_timestamp_examples=missing_examples,
        rows_by_series=rows_by_series,
        missing_timestamps_by_series=missing_timestamps_by_series,
    )
    return frame, report


def interpolate_short_gaps(
    frame: pd.DataFrame,
    expected_frequency: str = "1h",
    max_gap_hours: int = 5,
) -> pd.DataFrame:
    """Fill short internal OHLCV gaps with causal flat candles.

    Each synthesized candle uses the last observed close for all OHLC fields,
    sets volume to zero, and is marked ``is_interpolated=True``. Leading and
    trailing gaps are not generated because they have no prior observed close.
    Internal gaps longer than ``max_gap_hours`` remain missing and unfilled.
    """
    if max_gap_hours < 1:
        raise ValueError("max_gap_hours debe ser al menos 1.")
    if "timestamp" not in frame.columns:
        raise ValueError("El DataFrame debe incluir timestamp.")
    if frame.empty:
        raise ValueError("No se pueden interpolar datos vacíos.")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas OHLCV: {missing_columns}.")
    if frame.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Limpie primero los valores nulos del dataset antes de interpolar."
        )
    if frame["timestamp"].isna().any():
        raise ValueError("No se pueden interpolar timestamps nulos.")
    if frame["timestamp"].duplicated().any():
        raise ValueError(
            "La interpolación requiere una serie única, sin timestamps duplicados."
        )
    if (
        "is_interpolated" in frame
        and frame["is_interpolated"].astype(bool).any()
    ):
        raise ValueError("La entrada debe ser el dataset original, sin velas sintéticas.")

    observed = frame.copy()
    observed["timestamp"] = pd.to_datetime(observed["timestamp"], utc=True)
    if not observed["timestamp"].is_monotonic_increasing:
        raise ValueError("Ordene cronológicamente la serie antes de interpolarla.")
    frequency = pd.Timedelta(expected_frequency)
    if frequency <= pd.Timedelta(0):
        raise ValueError("expected_frequency debe ser positivo.")
    max_gap_steps = int(
        pd.Timedelta(hours=max_gap_hours) / frequency
    )
    if max_gap_steps < 1:
        raise ValueError("max_gap_hours debe cubrir al menos un intervalo.")

    full_index = pd.date_range(
        start=observed["timestamp"].iloc[0],
        end=observed["timestamp"].iloc[-1],
        freq=frequency,
    )
    indexed = observed.set_index("timestamp").reindex(full_index)
    indexed.index.name = "timestamp"
    missing = ~indexed.index.isin(pd.DatetimeIndex(observed["timestamp"]))
    missing_series = pd.Series(missing, index=indexed.index)
    gap_id = missing_series.ne(missing_series.shift(fill_value=False)).cumsum()
    gap_length = missing_series.groupby(gap_id).transform("sum")
    fillable = missing_series & gap_length.le(max_gap_steps)

    prior_close = indexed["close"].ffill()
    for column in ("open", "high", "low", "close"):
        indexed.loc[fillable, column] = prior_close.loc[fillable]
    indexed.loc[fillable, "volume"] = 0.0
    for column in ("exchange", "symbol"):
        if column in indexed.columns:
            indexed.loc[fillable, column] = indexed[column].ffill().loc[fillable]
    indexed["is_interpolated"] = fillable.astype(bool)
    return indexed.reset_index()


def combine_symbol_datasets(
    datasets: dict[tuple[str, str], str | Path],
    output_path: str | Path,
    expected_frequency: str = "1h",
    since: str | None = None,
    until_exclusive: str | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Combine exchange/symbol OHLCV files into one pickle dataset."""
    if not datasets:
        raise ValueError("Debe proporcionarse al menos un dataset por serie.")

    frames = []
    for (exchange, symbol), path in datasets.items():
        if not exchange.strip() or not symbol.strip():
            raise ValueError("Exchange y símbolo no pueden estar vacíos.")
        frame, _ = load_ohlcv(path, expected_frequency=expected_frequency)
        if "symbol" in frame.columns or "exchange" in frame.columns:
            raise ValueError(
                f"El archivo para {exchange} {symbol} ya incluye identificadores "
                "de serie que podrían ser sobrescritos."
            )
        frame["symbol"] = symbol
        frame["exchange"] = exchange
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["timestamp", "exchange", "symbol"], kind="stable"
    )
    if since is not None:
        start = pd.Timestamp(since)
        if start.tz is None:
            start = start.tz_localize("UTC")
        combined = combined.loc[combined["timestamp"] >= start.tz_convert("UTC")]
    if until_exclusive is not None:
        end = pd.Timestamp(until_exclusive)
        if end.tz is None:
            end = end.tz_localize("UTC")
        combined = combined.loc[combined["timestamp"] < end.tz_convert("UTC")]
    if combined.empty:
        raise ValueError("No hay datos dentro del rango temporal configurado.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(output)
    return load_ohlcv(output, expected_frequency=expected_frequency)


def load_split_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    """Read and validate the version-controlled temporal split boundaries."""
    config_path = Path(path)
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    split = config.get("split")
    required = (
        "train_end_exclusive",
        "validation_end_exclusive",
        "held_out_end_exclusive",
    )
    if not isinstance(split, dict) or any(key not in split for key in required):
        raise ValueError(f"Configuración de particiones incompleta en {config_path}.")

    parsed = {key: pd.Timestamp(split[key]) for key in required}
    if any(timestamp.tz is None for timestamp in parsed.values()):
        raise ValueError("Las fechas de corte deben incluir zona horaria (UTC).")
    if not (
        parsed[required[0]] < parsed[required[1]] < parsed[required[2]]
    ):
        raise ValueError("Las fechas de corte deben estar en orden cronológico.")
    return {key: parsed[key].tz_convert("UTC").isoformat() for key in required}


def split_by_dates(
    frame: pd.DataFrame,
    split_config: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Split rows into non-overlapping train, validation, and held-out periods.

    The intervals are [start, end), so observations on a cutoff belong to the
    partition beginning at that cutoff. Rows outside the configured range are
    intentionally excluded.
    """
    if "timestamp" not in frame.columns:
        raise ValueError("El DataFrame debe incluir la columna timestamp.")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("No se puede dividir: hay timestamps inválidos.")
    series_columns = [
        column for column in ("exchange", "symbol") if column in frame.columns
    ]
    if series_columns:
        chronology = pd.MultiIndex.from_frame(
            pd.concat(
                [timestamps.rename("timestamp"), frame[series_columns].reset_index(drop=True)],
                axis=1,
            )
        )
        if not chronology.is_monotonic_increasing:
            raise ValueError(
                "Ordene el dataset por timestamp, exchange y symbol antes de dividirlo."
            )
    elif not timestamps.is_monotonic_increasing:
        raise ValueError("Ordene el dataset cronológicamente antes de dividirlo.")

    train_end = pd.Timestamp(split_config["train_end_exclusive"])
    validation_end = pd.Timestamp(split_config["validation_end_exclusive"])
    held_out_end = pd.Timestamp(split_config["held_out_end_exclusive"])
    if any(value.tz is None for value in (train_end, validation_end, held_out_end)):
        raise ValueError("Las fechas de corte deben incluir zona horaria (UTC).")
    if not train_end < validation_end < held_out_end:
        raise ValueError("Las fechas de corte deben estar en orden cronológico.")

    partitions = {
        "train": frame.loc[timestamps < train_end].copy(),
        "validation": frame.loc[
            (timestamps >= train_end) & (timestamps < validation_end)
        ].copy(),
        "held_out": frame.loc[
            (timestamps >= validation_end) & (timestamps < held_out_end)
        ].copy(),
    }
    empty = [name for name, data in partitions.items() if data.empty]
    if empty:
        raise ValueError(
            f"Particiones vacías: {empty}. Verifique cobertura del dataset y fechas "
            "en configs/project.toml."
        )
    return partitions
