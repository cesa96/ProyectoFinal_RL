"""Download and validate the configured Binance OHLCV dataset."""

from __future__ import annotations

import datetime
import tempfile
import tomllib
from pathlib import Path

from gym_trading_env.downloader import download
import pandas as pd

from src.data import load_ohlcv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.toml"


def main() -> None:
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)

    dataset = config["dataset"]
    output_path = ROOT / dataset["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    since = datetime.datetime.fromisoformat(dataset["since"])
    until = datetime.datetime.fromisoformat(dataset["until_exclusive"])

    with tempfile.TemporaryDirectory(
        prefix="market-download-", dir=output_path.parent
    ) as temporary_directory:
        download(
            exchange_names=[dataset["exchange"]],
            symbols=dataset["symbols"],
            timeframe=dataset["timeframe"],
            dir=temporary_directory,
            since=since,
            until=until,
        )

        source_files = [
            Path(temporary_directory)
            / (
                f"{dataset['exchange']}-{symbol.replace('/', '')}-"
                f"{dataset['timeframe']}.pkl"
            )
            for symbol in dataset["symbols"]
        ]
        missing_files = [str(path) for path in source_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "El descargador no generó los archivos esperados: "
                + ", ".join(missing_files)
            )
        if len(source_files) != 1:
            raise ValueError(
                "Este proyecto está configurado para un único archivo y símbolo."
            )

        frame, _ = load_ohlcv(
            source_files[0],
            expected_frequency=dataset["expected_frequency"],
        )
        start = pd.Timestamp(dataset["since"], tz="UTC")
        end = pd.Timestamp(dataset["until_exclusive"], tz="UTC")
        frame = frame.loc[
            (frame["timestamp"] >= start) & (frame["timestamp"] < end)
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError("La descarga no contiene datos en el período solicitado.")
        frame.to_pickle(output_path)

    validated, report = load_ohlcv(
        output_path,
        expected_frequency=dataset["expected_frequency"],
    )
    print(f"Dataset Binance guardado en {output_path}")
    print(f"Filas: {len(validated):,}")
    print(report.to_dict())

    obsolete_dataset = output_path.parent / "BTCUSDT-1h-multi-exchange.pkl"
    if obsolete_dataset.is_file():
        obsolete_dataset.unlink()


if __name__ == "__main__":
    main()
