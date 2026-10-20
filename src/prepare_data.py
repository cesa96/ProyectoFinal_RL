"""Create a causal, gap-filled derivative of the original OHLCV dataset."""

from __future__ import annotations

import tomllib
from pathlib import Path

from src.data import interpolate_short_gaps, load_ohlcv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.toml"


def main() -> None:
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)

    dataset_config = config["dataset"]
    preprocessing = config["preprocessing"]
    if preprocessing["method"] != "causal_flat_forward_fill":
        raise ValueError(
            f"Método de imputación no soportado: {preprocessing['method']}"
        )
    source_path = ROOT / dataset_config["path"]
    output_path = ROOT / preprocessing["interpolated_path"]
    frequency = dataset_config["expected_frequency"]

    original, quality = load_ohlcv(source_path, expected_frequency=frequency)
    prepared = interpolate_short_gaps(
        original,
        expected_frequency=frequency,
        max_gap_hours=preprocessing["max_gap_hours"],
    )
    ohlcv_columns = ["open", "high", "low", "close", "volume"]
    remaining_nulls = int(
        prepared.loc[:, ohlcv_columns].isna().any(axis=1).sum()
    )
    if remaining_nulls:
        raise ValueError(
            f"Quedaron {remaining_nulls} timestamps sin completar. "
            "Revise los huecos largos antes de generar el dataset derivado."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_pickle(output_path)
    validated, processed_quality = load_ohlcv(
        output_path, expected_frequency=frequency
    )
    interpolated_rows = int(validated["is_interpolated"].sum())
    if len(validated) != len(prepared) or processed_quality.missing_timestamps:
        raise ValueError("La validación del dataset derivado no pasó.")
    if interpolated_rows != quality.missing_timestamps:
        raise ValueError(
            "El número de velas sintéticas no coincide con los huecos internos "
            "detectados. No se considera la ausencia inicial sin precio previo."
        )

    print(f"Dataset original conservado: {source_path}")
    print(f"Dataset derivado guardado: {output_path}")
    print(f"Filas originales: {len(original):,}")
    print(f"Velas sintéticas añadidas: {interpolated_rows:,}")
    print(f"Filas preparadas: {len(validated):,}")
    print("Huecos internos restantes: 0")


if __name__ == "__main__":
    main()
