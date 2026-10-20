"""Congela una selección establecida con validación antes de la prueba final."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from src.evaluate import RANDOM_SEEDS, resolve, sha256, write_json
from src.env_factory import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/v5_turnover_drawdown/seed_42")
    parser.add_argument("--destination", default="models/final")
    args = parser.parse_args(argv)
    run = resolve(args.run)
    destination = resolve(args.destination)
    # Nunca reemplazar una selección que ya pudo haberse evaluado.
    if destination.exists():
        raise FileExistsError(f"El destino ya existe: {destination}")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    dataset = PROJECT_ROOT / metadata["dataset"]["path"]
    if sha256(dataset) != metadata["dataset"]["sha256"]:
        raise ValueError("Los datos difieren de los usados en el entrenamiento.")
    destination.mkdir(parents=True)
    for original, target in [("best_model.zip", "model.zip"), ("config.json", "config.json"),
                             ("metadata.json", "training_metadata.json"),
                             ("validation_log.csv", "validation_log.csv")]:
        shutil.copy2(run / original, destination / target)
    files = sorted((PROJECT_ROOT / "src").glob("*.py"))
    files += sorted((PROJECT_ROOT / "configs").glob("*.toml"))
    files += [dataset, destination / "model.zip", destination / "config.json",
              destination / "training_metadata.json", PROJECT_ROOT / "pyproject.toml"]
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_run": run.relative_to(PROJECT_ROOT).as_posix(),
        "selection": ("Recomendación previa basada en validación: recompensa v5; "
                      "semilla 42 con retorno mediano entre 42, 43 y 44; "
                      "mejor checkpoint de esa corrida por retorno de validación."),
        "random_seeds": RANDOM_SEEDS,
        "official_partition": "held_out",
        "final_liquidation": False,
        "initial_position": 0,
        "model_selection_uses_final_results": False,
        "files_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in files},
    }
    write_json(destination / "PROTOCOL.json", manifest)
    print(f"Protocolo congelado en {destination}")


if __name__ == "__main__":
    main()
