"""Reporte de la iteración de la recompensa.

Complementa `python -m src.analysis compare` con las medidas robustas usadas
en el estudio de sensibilidad: el mejor checkpoint es el máximo
de ~15 evaluaciones ruidosas de validación y por eso es optimista. Aquí se
reportan además el último checkpoint, el promedio de la segunda mitad del
entrenamiento y cuántas evaluaciones superan a comprar y mantener.

    python -m src.reward_report

Solo lee resultados de validación ya guardados en runs/; no carga el held-out.
Salidas en runs/comparison/: reward_robust.md, reward_robust.csv y
reward_validation_curves.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import PROJECT_ROOT

REWARDS = [
    "v1_net_return",
    "v2_turnover_penalty",
    "v3_drawdown_penalty",
    "v4_drawdown_scaled",
    "v5_turnover_drawdown",
]
LATE_FROM = 150_000  # segunda mitad del presupuesto de 300.000 pasos
# Comprar y mantener en validación con el mismo entorno y costos
# (docs/algoritmo_y_entrenamiento.md, sección 4).
BUY_AND_HOLD_RETURN = 0.7933
BUY_AND_HOLD_DRAWDOWN = -0.309


def robust_table(runs_dir: Path, rewards: list[str]) -> pd.DataFrame:
    rows = []
    for reward in rewards:
        for log_path in sorted((runs_dir / reward).glob("seed_*/validation_log.csv")):
            log = pd.read_csv(log_path)
            late = log[log["timesteps"] >= LATE_FROM]
            best = log.loc[log["total_return"].idxmax()]
            rows.append({
                "reward": reward,
                "seed": int(log_path.parent.name.removeprefix("seed_")),
                "best_return": best["total_return"],
                "last_return": log["total_return"].iloc[-1],
                "late_mean_return": late["total_return"].mean(),
                "late_mean_drawdown": late["max_drawdown"].mean(),
                "late_mean_trades": late["n_trades"].mean(),
                "late_mean_cost": late["total_cost"].mean(),
                "evals_above_bh": int((log["total_return"] > BUY_AND_HOLD_RETURN).sum()),
                "n_evals": len(log),
            })
    return pd.DataFrame(rows)


def _pm(values: pd.Series, pct: bool = True) -> str:
    mean, std = values.mean(), values.std(ddof=1)
    return f"{mean:+.1%} ± {std:.1%}" if pct else f"{mean:,.0f} ± {std:,.0f}"


def markdown(table: pd.DataFrame, rewards: list[str]) -> str:
    lines = [
        "| Recompensa | Mejor checkpoint (optimista) | Último checkpoint | "
        "Promedio 2.ª mitad | Drawdown 2.ª mitad | Operaciones 2.ª mitad | "
        "Costos 2.ª mitad | Evaluaciones > comprar y mantener |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for reward in rewards:
        sub = table[table["reward"] == reward]
        lines.append(
            f"| {reward} | {_pm(sub['best_return'])} | {_pm(sub['last_return'])} | "
            f"{_pm(sub['late_mean_return'])} | {_pm(sub['late_mean_drawdown'])} | "
            f"{_pm(sub['late_mean_trades'], pct=False)} | "
            f"{_pm(sub['late_mean_cost'], pct=False)} | "
            f"{sub['evals_above_bh'].sum()} de {sub['n_evals'].sum()} |"
        )
    lines.append(
        f"| *Comprar y mantener* | *{BUY_AND_HOLD_RETURN:+.1%}* | | | "
        f"*{BUY_AND_HOLD_DRAWDOWN:+.1%}* | *1* | *100* | |"
    )
    return "\n".join(lines) + "\n"


def plot_curves(runs_dir: Path, rewards: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = plt.get_cmap("tab10")
    for i, reward in enumerate(rewards):
        logs = [pd.read_csv(p).set_index("timesteps")
                for p in sorted((runs_dir / reward).glob("seed_*/validation_log.csv"))]
        for column, ax in zip(("total_return", "n_trades"), axes):
            frame = pd.concat([log[column] for log in logs], axis=1)
            mean = frame.mean(axis=1)
            ax.plot(mean.index, mean.values, color=colors(i), label=reward, lw=1.6)
            ax.fill_between(mean.index, frame.min(axis=1), frame.max(axis=1),
                            color=colors(i), alpha=0.12, lw=0)
    axes[0].axhline(BUY_AND_HOLD_RETURN, color="grey", ls="--", label="Comprar y mantener")
    axes[0].set_title("Retorno en validación (media de 3 semillas; banda: mín–máx)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Operaciones en validación")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Pasos de entrenamiento")
        ax.axvline(LATE_FROM, color="black", lw=0.6, ls=":")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rewards", nargs="*", default=REWARDS)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args(argv)

    runs_dir = PROJECT_ROOT / args.runs_dir
    out = runs_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    table = robust_table(runs_dir, args.rewards)
    table.to_csv(out / "reward_robust.csv", index=False)
    text = markdown(table, args.rewards)
    (out / "reward_robust.md").write_text(text, encoding="utf-8")
    plot_curves(runs_dir, args.rewards, out / "reward_validation_curves.png")
    print(text)
    print(f"Archivos en {out}")


if __name__ == "__main__":
    sys.exit(main())
