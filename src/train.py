"""Entrenamiento del agente DQN.

Uso
---
Entrenar la recompensa configurada con todas las semillas de configs/dqn.toml:

    python -m src.train

Entrenar otra versión de la recompensa (definida en
src/rewards.py) o cambiar semillas / pasos sin editar el archivo:

    python -m src.train --reward v2_mi_recompensa
    python -m src.train --seeds 42 43 44 --timesteps 300000

Congelar el agente final que eligió el equipo (con validación) para entregarlo
a la evaluación final:

    python -m src.train --freeze runs/v1_net_return/seed_42

Protocolo
---------
1. Solo se construyen los entornos de entrenamiento y validación. El held-out
   no se carga en este script en ningún momento.
2. Entrenamiento: episodios de `train_episode_length` horas con inicio
   aleatorio dentro del periodo de entrenamiento (el inicio depende de la
   semilla del entorno, por lo que es reproducible).
3. Cada `eval_freq` pasos se recorre la partición de validación completa con la
   política determinista (sin exploración). Si la métrica de validación mejora,
   se guarda el checkpoint como best_model.zip.
4. Criterio de parada: se alcanza `total_timesteps` o la métrica de validación
   no mejora durante `patience` evaluaciones seguidas.
5. Se repite con cada semilla y se reporta media y desviación entre semillas.

Salidas en runs/<recompensa>/seed_<semilla>/:
    best_model.zip        mejor checkpoint según validación
    last_model.zip        modelo al terminar el entrenamiento
    monitor.csv           recompensa y longitud de cada episodio de entrenamiento
    validation_log.csv    métricas de validación en cada evaluación
    config.json           configuración efectiva usada (incluye overrides)
    metadata.json         semilla, commit, versiones, hash del dataset, tiempos
y en runs/<recompensa>/: summary.csv y learning_curves.png.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import (
    DEFAULT_DQN_CONFIG,
    PROJECT_ROOT,
    DQNAgent,
    build_dqn,
    load_dqn_config,
    run_episode,
)
from src.data import DEFAULT_CONFIG_PATH
from src.env_factory import build_market_data
from src.environment import TradingEnvironment
from src.rewards import make_reward

SELECTION_METRICS = ("total_return", "calmar")


# ----------------------------------------------------------------------
# Entornos (solo entrenamiento y validación)
# ----------------------------------------------------------------------
def make_train_and_validation_envs(config: dict, monitor_path: Path | None = None):
    """Crea el entorno de entrenamiento y el de validación.

    build_market_data calcula los indicadores sobre la serie completa (todas las
    ventanas miran hacia atrás) y ajusta el escalador solo con entrenamiento.
    Aquí se descarta explícitamente el held-out para que nada de este script
    pueda usarlo.
    """
    from stable_baselines3.common.monitor import Monitor

    data, _ = build_market_data(DEFAULT_CONFIG_PATH)
    data.pop("held_out")  # regla de oro: el held-out no se usa al entrenar

    env_cfg = config["env"]
    reward_name = config["training"]["reward"]
    common = {
        "initial_capital": env_cfg["initial_capital"],
        "transaction_cost_bps": env_cfg["transaction_cost_bps"],
    }

    train = data["train"]
    train_env = TradingEnvironment(
        prices=train.prices,
        features=train.features,
        timestamps=train.timestamps,
        reward_fn=make_reward(reward_name),
        episode_length=env_cfg["train_episode_length"],
        random_start=env_cfg["train_random_start"],
        **common,
    )
    if monitor_path is not None:
        train_env = Monitor(train_env, filename=str(monitor_path))

    val = data["validation"]
    validation_env = TradingEnvironment(
        prices=val.prices,
        features=val.features,
        timestamps=val.timestamps,
        reward_fn=make_reward(reward_name),
        record_history=True,
        **common,
    )
    return train_env, validation_env


# ----------------------------------------------------------------------
# Evaluación periódica en validación + early stopping
# ----------------------------------------------------------------------
def _make_validation_callback(validation_env, run_dir: Path, metric: str,
                              eval_freq: int, patience: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationCallback(BaseCallback):
        """Evalúa en validación, guarda el mejor checkpoint y aplica patience."""

        def __init__(self):
            super().__init__()
            self.best_score = -np.inf
            self.best_timestep = 0
            self.evals_without_improvement = 0
            self.rows: list[dict] = []
            self.last_eval_timestep = -1
            self.stopped_early = False

        def _evaluate(self) -> bool:
            agent = DQNAgent(self.model)
            summary = run_episode(validation_env, agent.act)
            score = summary[metric]
            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.best_timestep = self.num_timesteps
                self.evals_without_improvement = 0
                self.model.save(run_dir / "best_model.zip")
            else:
                self.evals_without_improvement += 1
            row = {
                "timesteps": self.num_timesteps,
                "epsilon": float(self.model.exploration_rate),
                **summary,
                "is_best": improved,
            }
            self.rows.append(row)
            pd.DataFrame(self.rows).to_csv(run_dir / "validation_log.csv", index=False)
            self.last_eval_timestep = self.num_timesteps
            print(
                f"  [{self.num_timesteps:>8} pasos] validación {metric}="
                f"{score:+.4f} retorno={summary['total_return']:+.2%} "
                f"maxDD={summary['max_drawdown']:.2%} trades={summary['n_trades']}"
                f"{'  <- mejor' if improved else ''}",
                flush=True,
            )
            if patience > 0 and self.evals_without_improvement >= patience:
                self.stopped_early = True
                print(f"  Early stopping: {patience} evaluaciones sin mejora.")
                return False
            return True

        def _on_step(self) -> bool:
            if self.num_timesteps % eval_freq == 0:
                return self._evaluate()
            return True

        def _on_training_end(self) -> None:
            if self.last_eval_timestep != self.num_timesteps and not self.stopped_early:
                self._evaluate()

    return ValidationCallback()


# ----------------------------------------------------------------------
# Trazabilidad
# ----------------------------------------------------------------------
def _git_commit() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }


def _dataset_path() -> Path:
    with DEFAULT_CONFIG_PATH.open("rb") as file:
        project_cfg = tomllib.load(file)
    return PROJECT_ROOT / project_cfg["preprocessing"]["interpolated_path"]


# ----------------------------------------------------------------------
# Entrenamiento de una semilla
# ----------------------------------------------------------------------
def train_one_seed(config: dict, seed: int, reward_dir: Path) -> dict:
    run_dir = reward_dir / f"seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    selection = config["selection"]
    metric = selection["metric"]
    if metric not in SELECTION_METRICS:
        raise ValueError(f"metric debe ser uno de {SELECTION_METRICS}.")

    train_env, validation_env = make_train_and_validation_envs(
        config, monitor_path=run_dir / "monitor.csv"
    )
    model = build_dqn(train_env, config, seed=seed)
    callback = _make_validation_callback(
        validation_env, run_dir, metric,
        eval_freq=selection["eval_freq"], patience=selection["patience"],
    )

    print(f"\n=== Recompensa {config['training']['reward']} | semilla {seed} ===")
    started = time.time()
    model.learn(total_timesteps=config["training"]["total_timesteps"], callback=callback)
    elapsed = time.time() - started
    model.save(run_dir / "last_model.zip")

    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)
    best_row = next(r for r in reversed(callback.rows) if r["is_best"])
    metadata = {
        "seed": seed,
        "reward": config["training"]["reward"],
        "selection_metric": metric,
        "best_timestep": callback.best_timestep,
        "best_validation": best_row,
        "timesteps_trained": int(model.num_timesteps),
        "stopped_early": callback.stopped_early,
        "elapsed_seconds": round(elapsed, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_commit(),
        "versions": _versions(),
        "dataset": {"path": str(_dataset_path().relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(_dataset_path())},
        "held_out_used": False,
    }
    with (run_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2, default=str)
    return {"seed": seed, "best_timestep": callback.best_timestep,
            "timesteps_trained": int(model.num_timesteps), **best_row}


# ----------------------------------------------------------------------
# Resumen entre semillas y curvas de aprendizaje
# ----------------------------------------------------------------------
def validation_buy_and_hold(config: dict) -> dict:
    """Comprar y mantener en validación, como referencia para las curvas.

    Usa el mismo entorno y costos que el agente (acción 100 % en todo momento).
    Solo es una línea de referencia del periodo de validación; la comparación
    oficial contra baselines se ejecuta con src.evaluate.
    """
    _, validation_env = make_train_and_validation_envs(config)
    full = len(validation_env.fractions) - 1
    return run_episode(validation_env, lambda obs: full)


def write_summary(results: list[dict], reward_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(results).drop(columns=["is_best"], errors="ignore")
    numeric = frame.drop(columns=["seed"]).select_dtypes("number")
    stats = pd.DataFrame([numeric.mean(), numeric.std(ddof=1)], index=["mean", "std"])
    stats.insert(0, "seed", stats.index)
    out = pd.concat([frame, stats], ignore_index=True)
    out.to_csv(reward_dir / "summary.csv", index=False)
    return out


def plot_learning_curves(reward_dir: Path, reference: dict | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = sorted(reward_dir.glob("seed_*"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for run_dir in run_dirs:
        label = run_dir.name.replace("_", " ")
        monitor = pd.read_csv(run_dir / "monitor.csv", skiprows=1)
        steps = monitor["l"].cumsum()
        smooth = monitor["r"].rolling(20, min_periods=1).mean()
        axes[0].plot(steps, smooth, label=label, lw=1.2)
        val = pd.read_csv(run_dir / "validation_log.csv")
        axes[1].plot(val["timesteps"], val["total_return"] * 100, marker="o", ms=3,
                     label=label, lw=1.2)
    axes[0].set_title("Entrenamiento: recompensa por episodio (media móvil 20)")
    axes[0].set_xlabel("Pasos de entrenamiento")
    axes[0].set_ylabel("Recompensa acumulada del episodio")
    axes[1].set_title("Validación: retorno acumulado de la política greedy")
    axes[1].set_xlabel("Pasos de entrenamiento")
    axes[1].set_ylabel("Retorno en validación (%)")
    if reference is not None:
        axes[1].axhline(reference["total_return"] * 100, color="grey", ls="--",
                        lw=1, label="Comprar y mantener (validación)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Curvas de aprendizaje — {reward_dir.name}")
    fig.tight_layout()
    path = reward_dir / "learning_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Congelar el agente final
# ----------------------------------------------------------------------
def freeze(run_dir: Path, destination: Path, force: bool = False) -> Path:
    """Copia el mejor checkpoint de una corrida a models/final/.

    La elección de qué corrida congelar se hace con validación (summary.csv);
    después de congelar no se reentrena ni se cambia nada antes del held-out.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "best_model.zip").exists():
        raise FileNotFoundError(f"No hay best_model.zip en {run_dir}.")
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} ya existe. El agente final solo se congela una vez; "
                "use --force únicamente si el equipo lo decidió explícitamente."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(run_dir / "best_model.zip", destination / "model.zip")
    for name in ("config.json", "metadata.json", "validation_log.csv"):
        shutil.copy2(run_dir / name, destination / name)
    frozen = {
        "source_run": str(run_dir.resolve().relative_to(PROJECT_ROOT))
        if run_dir.resolve().is_relative_to(PROJECT_ROOT) else str(run_dir),
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_commit(),
        "model_sha256": _sha256(destination / "model.zip"),
    }
    with (destination / "FROZEN.json").open("w") as file:
        json.dump(frozen, file, indent=2)
    print(f"Agente congelado en {destination}. Haga commit de esta carpeta "
          "antes de la evaluación en held-out.")
    return destination


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Entrenamiento DQN")
    parser.add_argument("--config", type=Path, default=DEFAULT_DQN_CONFIG)
    parser.add_argument("--reward", help="nombre de la recompensa en src/rewards.py")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--freeze", type=Path, metavar="RUN_DIR",
                        help="congela el best_model.zip de esa corrida en models/final")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECCION.CLAVE=VALOR",
                        help="cambia un valor de configs/dqn.toml sin editarlo, "
                             "p. ej. --set agent.gamma=0.95 (se puede repetir)")
    return parser.parse_args(argv)


def apply_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    training = config["training"]
    if args.reward:
        training["reward"] = args.reward
    if args.seeds:
        training["seeds"] = args.seeds
    if args.timesteps:
        training["total_timesteps"] = args.timesteps
    if args.eval_freq:
        config["selection"]["eval_freq"] = args.eval_freq
    if args.output_dir:
        training["output_dir"] = str(args.output_dir)
    for assignment in args.set:
        key, _, raw = assignment.partition("=")
        section, _, name = key.partition(".")
        if not raw or section not in config or name not in config[section]:
            raise ValueError(f"--set inválido: {assignment!r}. Use SECCION.CLAVE=VALOR "
                             "con una clave que exista en configs/dqn.toml.")
        current = config[section][name]
        config[section][name] = type(current)(json.loads(raw)) if not isinstance(
            current, str) else raw
    return config


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.freeze:
        freeze(args.freeze, PROJECT_ROOT / "models" / "final", force=args.force)
        return

    config = apply_overrides(load_dqn_config(args.config), args)
    make_reward(config["training"]["reward"])  # falla rápido si el nombre no existe
    output_dir = Path(config["training"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    reward_dir = output_dir / config["training"]["reward"]
    reward_dir.mkdir(parents=True, exist_ok=True)

    results = [train_one_seed(config, seed, reward_dir) for seed in config["training"]["seeds"]]
    summary = write_summary(results, reward_dir)
    reference = validation_buy_and_hold(config)
    figure = plot_learning_curves(reward_dir, reference)

    cols = ["seed", "best_timestep", "total_return", "max_drawdown", "total_cost",
            "n_trades", "pct_time_0", "pct_time_50", "pct_time_100"]
    print("\nResumen en validación (mejor checkpoint de cada semilla):")
    print(summary[cols].to_string(index=False))
    print(f"Referencia comprar y mantener en validación: "
          f"retorno={reference['total_return']:+.2%} maxDD={reference['max_drawdown']:.2%}")
    print(f"Curvas: {figure}")


if __name__ == "__main__":
    sys.exit(main())
