"""Sensibilidad del agente DQN a sus hiperparámetros (Persona 4).

Se cambia UN hiperparámetro a la vez respecto a la configuración inicial
(γ = 0,99, learning rate = 1e-4), con la recompensa v1, las mismas 3 semillas y el mismo
protocolo. Todo se mide en validación; el held-out no se usa.

    python -m src.sensitivity run                 # entrena todas las variantes
    python -m src.sensitivity run --only gamma_0.95
    python -m src.sensitivity summarize           # tabla y gráfico

Salidas en runs/sensitivity/: una carpeta por variante, sensitivity.md
(tabla para el informe), sensitivity.csv y sensitivity.png.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.agent import PROJECT_ROOT
from src.train import main as train_main

REWARD = "v1_net_return"
OUTPUT = PROJECT_ROOT / "runs" / "sensitivity"

# nombre -> (descripción, valores usados). Cada variante declara γ y learning
# rate de forma explícita, de modo que se reproduce igual aunque cambie la
# configuración base. Se cambia un hiperparámetro a la vez respecto a la
# configuración inicial (γ = 0,99, lr = 1e-4). El resto viene de configs/dqn.toml.
INITIAL = "lr_1e-4"
CHOSEN = "lr_3e-4"
VARIANTS: dict[str, tuple[str, list[str]]] = {
    "lr_1e-4": ("Inicial: γ = 0,99, lr = 1×10⁻⁴",
                ["agent.gamma=0.99", "agent.learning_rate=1e-4"]),
    "gamma_0.95": ("γ = 0,95 (horizonte ~20 h), lr = 1×10⁻⁴",
                   ["agent.gamma=0.95", "agent.learning_rate=1e-4"]),
    "gamma_0.999": ("γ = 0,999 (horizonte ~1.000 h), lr = 1×10⁻⁴",
                    ["agent.gamma=0.999", "agent.learning_rate=1e-4"]),
    "lr_3e-5": ("γ = 0,99, lr = 3×10⁻⁵",
                ["agent.gamma=0.99", "agent.learning_rate=3e-5"]),
    "lr_3e-4": ("γ = 0,99, lr = 3×10⁻⁴ (elegida)",
                ["agent.gamma=0.99", "agent.learning_rate=3e-4"]),
}


def run(only: list[str] | None = None) -> None:
    for name in only or VARIANTS:
        _, changes = VARIANTS[name]
        args = ["--reward", REWARD, "--output-dir", str(OUTPUT / name)]
        for change in changes:
            args += ["--set", change]
        print(f"\n##### Variante {name}: {', '.join(changes)}")
        train_main(args)


def _seed_rows(summary_path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    return summary[~summary["seed"].astype(str).isin(["mean", "std"])].astype(
        {"total_return": float, "max_drawdown": float, "total_cost": float,
         "n_trades": float, "pct_time_100": float})


LATE_FROM = 150_000  # "segunda mitad" del entrenamiento (de 300.000 pasos)


def _robust(reward_dir) -> dict:
    """Medidas que no dependen de quedarse con el mejor pico de validación.

    El mejor checkpoint es el máximo de ~15 evaluaciones ruidosas, así que una
    variante más inestable obtiene picos más altos por azar. El último
    checkpoint y el promedio de la segunda mitad del entrenamiento son más
    justos para comparar variantes.
    """
    last, late, late_trades = [], [], []
    for log_path in sorted(reward_dir.glob("seed_*/validation_log.csv")):
        log = pd.read_csv(log_path)
        last.append(log["total_return"].iloc[-1])
        second_half = log[log["timesteps"] >= LATE_FROM]
        late.append(second_half["total_return"].mean())
        late_trades.append(second_half["n_trades"].mean())
    series = pd.Series
    return {
        "last_return_mean": series(last).mean(),
        "late_return_mean": series(late).mean(),
        "late_return_std": series(late).std(ddof=1),
        "late_trades_mean": series(late_trades).mean(),
    }


def summarize() -> pd.DataFrame:
    entries = [(name, desc, OUTPUT / name / REWARD / "summary.csv")
               for name, (desc, _) in VARIANTS.items()]

    rows = []
    for name, desc, path in entries:
        if not path.exists():
            print(f"Falta {path}; ejecute: python -m src.sensitivity run --only {name}")
            continue
        seeds = _seed_rows(path)
        robust = _robust(path.parent)
        rows.append({
            "variant": name, "description": desc, "n_seeds": len(seeds),
            "return_mean": seeds["total_return"].mean(),
            "return_std": seeds["total_return"].std(ddof=1),
            "return_min": seeds["total_return"].min(),
            "return_max": seeds["total_return"].max(),
            "drawdown_mean": seeds["max_drawdown"].mean(),
            "cost_mean": seeds["total_cost"].mean(),
            "trades_mean": seeds["n_trades"].mean(),
            "pct_100_mean": seeds["pct_time_100"].mean(),
            **robust,
        })
    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "sensitivity.csv", index=False)
    markdown = _markdown(table)
    (OUTPUT / "sensitivity.md").write_text(markdown, encoding="utf-8")
    _plot(table)
    print(markdown)
    return table


def _markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Variante | Promedio 2.ª mitad (media ± desv.) | Último checkpoint "
        "| Operaciones 2.ª mitad | Mejor checkpoint (optimista) | Máx. drawdown* "
        "| Costos* | % tiempo 100 %* |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.description} | {r.late_return_mean:+.1%} ± {r.late_return_std:.1%} "
            f"| {r.last_return_mean:+.1%} | {r.late_trades_mean:,.0f} "
            f"| {r.return_mean:+.1%} ± {r.return_std:.1%} | {r.drawdown_mean:.1%} "
            f"| {r.cost_mean:,.0f} | {r.pct_100_mean:.0%} |"
        )
    lines.append("")
    lines.append("Retornos en validación, media entre 3 semillas. «Promedio 2.ª mitad»: "
                 f"promedio de las evaluaciones desde el paso {LATE_FROM:,}. "
                 "* Medido en el mejor checkpoint de cada semilla.")
    return "\n".join(lines) + "\n"


def _plot(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    x = range(len(table))
    colors = ["#e76f51" if v == CHOSEN else "#2a9d8f" for v in table["variant"]]
    axes[0].bar(x, table["late_return_mean"] * 100, yerr=table["late_return_std"] * 100,
                capsize=4, color=colors)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Retorno en validación (%), promedio 2.ª mitad del entrenamiento")
    axes[1].bar(x, table["late_trades_mean"], color=colors)
    axes[1].set_title("Operaciones en validación, promedio 2.ª mitad")
    for ax in axes:
        ax.set_xticks(list(x), table["variant"], rotation=15, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Sensibilidad a hiperparámetros (recompensa v1, 3 semillas)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "sensitivity.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Sensibilidad a hiperparámetros")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--only", nargs="+", choices=list(VARIANTS))
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.only)
    else:
        summarize()


if __name__ == "__main__":
    sys.exit(main())
