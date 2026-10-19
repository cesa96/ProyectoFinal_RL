"""Herramientas para analizar el comportamiento del agente (apoyo a la Persona 5).

Permiten diagnosticar y comparar versiones de la recompensa sin escribir código
nuevo. Solo trabajan con las particiones de entrenamiento y validación: el
held-out está bloqueado aquí, porque la elección de la recompensa se hace con
validación (regla de oro del proyecto).

1) Ver qué hace un agente entrenado, paso a paso:

    python -m src.analysis inspect runs/v1_net_return/seed_44

   Guarda en runs/v1_net_return/seed_44/analysis/:
     trajectory.csv   una fila por hora: precio, posición, valor, costo,
                      recompensa y valores Q de cada acción
     behavior.png     precio con la posición sombreada, curva de capital frente
                      a comprar y mantener, y operaciones acumuladas
     stats.json       métricas del recorrido (retorno, drawdown, costos,
                      operaciones, % de tiempo en cada posición, duración media
                      de cada posición)

   Opciones: --model last (último modelo en vez del mejor),
             --partition train (recorrer entrenamiento en vez de validación),
             --start 2025-03-01 --end 2025-04-01 (acercar el gráfico a un
             periodo para mostrar una anomalía).

2) Comparar versiones de la recompensa (todas las semillas, en validación):

    python -m src.analysis compare v1_net_return v2_mi_recompensa

   Guarda en runs/comparison/: reward_comparison.csv, reward_comparison.md
   (tabla lista para el informe) y reward_comparison.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import PROJECT_ROOT, DQNAgent, summarize_history
from src.train import make_train_and_validation_envs

ALLOWED_PARTITIONS = ("train", "validation")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ----------------------------------------------------------------------
# Recorrido detallado de una política
# ----------------------------------------------------------------------
def rollout(run_dir: str | Path, model: str = "best",
            partition: str = "validation") -> tuple[pd.DataFrame, dict]:
    """Recorre una partición completa con la política greedy de una corrida.

    Usa la misma configuración (recompensa, costos, capital) con la que se
    entrenó la corrida, leída de su config.json.
    """
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(
            f"partition debe ser una de {ALLOWED_PARTITIONS}. El held-out no se "
            "usa para analizar ni para elegir la recompensa."
        )
    run_dir = _resolve(run_dir)
    with (run_dir / "config.json").open() as file:
        config = json.load(file)
    agent = DQNAgent.load(run_dir / f"{model}_model.zip")

    train_env, validation_env = make_train_and_validation_envs(config)
    if partition == "validation":
        env = validation_env
    else:
        # Recorre todo el periodo de entrenamiento desde el inicio, sin cortes.
        from src.environment import TradingEnvironment

        base = train_env
        env = TradingEnvironment(
            prices=base.prices, features=base.features, timestamps=base.timestamps,
            reward_fn=base.reward_fn, initial_capital=base.initial_capital,
            transaction_cost_bps=base.cost_rate / 1e-4, record_history=True,
        )

    obs, _ = env.reset()
    q_rows, done, total_reward = [], False, 0.0
    while not done:
        q_rows.append(agent.q_values(obs))
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        total_reward += reward
        done = terminated or truncated

    history = env.get_history()
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(history["timestamp"], utc=True),
        "price": history["price"],
        "position": history["position"],
        "portfolio_value": history["portfolio_value"],
        "transaction_cost": history["transaction_cost"],
        "reward": history["reward"],
    })
    q_values = np.vstack(q_rows)
    for i, fraction in enumerate(env.fractions):
        frame[f"q_{int(round(fraction * 100))}"] = q_values[:, i]
    frame["traded"] = frame["position"].diff().fillna(frame["position"]).ne(0)
    # Comprar y mantener con el mismo costo de entrada, como referencia.
    entry_cost = env.cost_rate * env.initial_capital
    next_prices = env.prices[1:len(frame) + 1]
    frame["buy_and_hold_value"] = (env.initial_capital - entry_cost) * (
        next_prices / env.prices[0]
    )

    stats = summarize_history(history, total_reward, env.fractions)
    runs = (frame["position"] != frame["position"].shift()).cumsum()
    stats["mean_hours_per_position"] = float(frame.groupby(runs).size().mean())
    stats["buy_and_hold_return"] = float(frame["buy_and_hold_value"].iloc[-1]
                                         / env.initial_capital - 1)
    stats.update({"run": str(run_dir.relative_to(PROJECT_ROOT))
                  if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
                  "model": model, "partition": partition,
                  "reward": config["training"]["reward"]})
    return frame, stats


def plot_behavior(frame: pd.DataFrame, stats: dict, path: Path,
                  start: str | None = None, end: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = frame
    if start:
        view = view[view["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        view = view[view["timestamp"] < pd.Timestamp(end, tz="UTC")]
    if view.empty:
        raise ValueError("No hay datos en el rango de fechas pedido.")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.3]})
    ts = view["timestamp"]

    ax = axes[0]
    ax.plot(ts, view["price"], color="black", lw=0.8, label="Precio BTC/USDT")
    colors = {0.5: ("#f2b134", "50 % invertido"), 1.0: ("#2a9d8f", "100 % invertido")}
    low, high = view["price"].min(), view["price"].max()
    for level, (color, label) in colors.items():
        ax.fill_between(ts, low, high, where=np.isclose(view["position"], level),
                        color=color, alpha=0.25, step="post", label=label, lw=0)
    ax.set_ylabel("Precio")
    ax.set_title(
        f"{stats['reward']} — {stats['run']} ({stats['model']} model, {stats['partition']})\n"
        "Sombreado: posición del agente (blanco = 0 %, en efectivo)"
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(ts, view["portfolio_value"], label="Agente DQN", color="#264653")
    ax.plot(ts, view["buy_and_hold_value"], label="Comprar y mantener",
            color="grey", ls="--")
    ax.set_ylabel("Valor del portafolio")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(ts, view["traded"].cumsum(), color="#e76f51")
    ax.set_ylabel("Operaciones\nacumuladas")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def inspect(run_dir: str | Path, model: str = "best", partition: str = "validation",
            start: str | None = None, end: str | None = None) -> Path:
    frame, stats = rollout(run_dir, model=model, partition=partition)
    out = _resolve(run_dir) / "analysis"
    out.mkdir(exist_ok=True)
    suffix = "" if (model, partition) == ("best", "validation") else f"_{model}_{partition}"
    frame.to_csv(out / f"trajectory{suffix}.csv", index=False)
    with (out / f"stats{suffix}.json").open("w") as file:
        json.dump(stats, file, indent=2)
    zoom = f"_{start or 'inicio'}_{end or 'fin'}" if (start or end) else ""
    figure = plot_behavior(frame, stats, out / f"behavior{suffix}{zoom}.png", start, end)
    print(json.dumps({k: stats[k] for k in (
        "total_return", "buy_and_hold_return", "max_drawdown", "total_cost",
        "n_trades", "mean_hours_per_position", "pct_time_0", "pct_time_50",
        "pct_time_100")}, indent=2))
    print(f"Archivos en {out}")
    return figure


# ----------------------------------------------------------------------
# Comparación entre versiones de la recompensa
# ----------------------------------------------------------------------
COMPARE_COLUMNS = {
    "total_return": "Retorno acumulado",
    "max_drawdown": "Máximo drawdown",
    "total_cost": "Costos pagados",
    "n_trades": "Operaciones",
    "pct_time_0": "% tiempo 0 %",
    "pct_time_50": "% tiempo 50 %",
    "pct_time_100": "% tiempo 100 %",
}


def compare(rewards: list[str], runs_dir: str | Path = "runs") -> pd.DataFrame:
    runs_dir = _resolve(runs_dir)
    rows = []
    for reward in rewards:
        summary_path = runs_dir / reward / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No existe {summary_path}. Entrene primero: "
                f"python -m src.train --reward {reward}"
            )
        summary = pd.read_csv(summary_path)
        seeds = summary[~summary["seed"].astype(str).isin(["mean", "std"])]
        for column in COMPARE_COLUMNS:
            values = seeds[column].astype(float)
            rows.append({"reward": reward, "metric": column,
                         "mean": values.mean(), "std": values.std(ddof=1),
                         "min": values.min(), "max": values.max(),
                         "n_seeds": len(values)})
    _warn_if_protocols_differ(rewards, runs_dir)
    table = pd.DataFrame(rows)
    out = runs_dir / "comparison"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "reward_comparison.csv", index=False)
    (out / "reward_comparison.md").write_text(_markdown_table(table, rewards),
                                              encoding="utf-8")
    _plot_comparison(table, rewards, out / "reward_comparison.png")
    print(_markdown_table(table, rewards))
    print(f"Archivos en {out}")
    return table


def _protocol(reward_dir: Path) -> dict:
    """Todo lo que debe ser igual entre versiones para que la comparación sea justa."""
    seeds, protocol = [], None
    for config_path in sorted(reward_dir.glob("seed_*/config.json")):
        with config_path.open() as file:
            config = json.load(file)
        seeds.append(int(config_path.parent.name.removeprefix("seed_")))
        protocol = {
            "agent": config["agent"], "exploration": config["exploration"],
            "env": config["env"], "selection": config["selection"],
            "total_timesteps": config["training"]["total_timesteps"],
        }
    return {"seeds": seeds, **(protocol or {})}


def _warn_if_protocols_differ(rewards: list[str], runs_dir: Path) -> None:
    reference = _protocol(runs_dir / rewards[0])
    for reward in rewards[1:]:
        other = _protocol(runs_dir / reward)
        differences = [key for key in reference if reference[key] != other.get(key)]
        if differences:
            print(
                f"ADVERTENCIA: {reward} y {rewards[0]} no se entrenaron con el mismo "
                f"protocolo (difieren: {', '.join(differences)}). La diferencia en "
                "resultados no se puede atribuir solo a la recompensa.\n"
            )


def _fmt(metric: str, mean: float, std: float) -> str:
    if metric in ("total_return", "max_drawdown") or metric.startswith("pct_time"):
        return f"{mean:+.1%} ± {std:.1%}" if not metric.startswith("pct") else f"{mean:.0%} ± {std:.0%}"
    return f"{mean:,.0f} ± {std:,.0f}"


def _markdown_table(table: pd.DataFrame, rewards: list[str]) -> str:
    header = "| Métrica (validación, media ± desv. entre semillas) | " + " | ".join(rewards) + " |"
    lines = [header, "|" + "---|" * (len(rewards) + 1)]
    for metric, label in COMPARE_COLUMNS.items():
        cells = []
        for reward in rewards:
            row = table[(table["reward"] == reward) & (table["metric"] == metric)].iloc[0]
            cells.append(_fmt(metric, row["mean"], row["std"]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _plot_comparison(table: pd.DataFrame, rewards: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["total_return", "max_drawdown", "total_cost", "n_trades"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.6))
    for ax, metric in zip(axes, metrics):
        sub = table[table["metric"] == metric].set_index("reward").loc[rewards]
        ax.bar(range(len(rewards)), sub["mean"], yerr=sub["std"], capsize=4,
               color="#2a9d8f")
        ax.set_xticks(range(len(rewards)), rewards, rotation=20, fontsize=8)
        ax.set_title(COMPARE_COLUMNS[metric])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Comparación de recompensas en validación (barras: media; líneas: desv. entre semillas)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Análisis del comportamiento del agente")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="recorrido detallado de una corrida")
    p_inspect.add_argument("run_dir")
    p_inspect.add_argument("--model", choices=["best", "last"], default="best")
    p_inspect.add_argument("--partition", choices=ALLOWED_PARTITIONS, default="validation")
    p_inspect.add_argument("--start")
    p_inspect.add_argument("--end")

    p_compare = sub.add_parser("compare", help="compara versiones de la recompensa")
    p_compare.add_argument("rewards", nargs="+")
    p_compare.add_argument("--runs-dir", default="runs")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        inspect(args.run_dir, args.model, args.partition, args.start, args.end)
    else:
        compare(args.rewards, args.runs_dir)


if __name__ == "__main__":
    sys.exit(main())
