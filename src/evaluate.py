"""Evaluación comparable y trazable de un modelo congelado y sus baselines."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd

from src.baselines import BuyAndHold, RandomAgent
from src.environment import TradingEnvironment
from src.env_factory import PROJECT_ROOT, build_market_data
from src.metrics import aggregate_random, financial_metrics
from src.rewards import make_reward

RANDOM_SEEDS = list(range(42, 62))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rollout(part, policy, config: dict, seed: int = 42):
    """Guarda por separado la hora de decisión t y la valoración t+1."""
    env = TradingEnvironment(
        part.prices, part.features, timestamps=part.timestamps,
        initial_capital=config["env"]["initial_capital"],
        transaction_cost_bps=config["env"]["transaction_cost_bps"],
        reward_fn=make_reward(config["training"]["reward"]),
        random_start=False, record_history=False,
    )
    obs, _ = env.reset(seed=seed)
    rows = []
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        i = info["index"]
        rows.append({
            "decision_time": part.timestamps[i - 1],
            "valuation_time": part.timestamps[i],
            "action": action,
            "position": info["position"],
            "price": info["price"],
            "next_price": info["next_price"],
            "portfolio_value": info["portfolio_value"],
            "transaction_cost": info["transaction_cost"],
            "turnover": info["turnover"],
            "net_return": info["net_return"],
            "drawdown": info["drawdown"],
            "reward": reward,
        })
        if terminated or truncated:
            break
    frame = pd.DataFrame(rows)
    metrics = financial_metrics(frame, env.initial_capital)
    env.close()
    return frame, metrics


def plot_results(trajectories, part, capital, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    dates = pd.to_datetime(part.timestamps)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, label, color in [
        ("rl", "Agente DQN", "#1c5369"),
        ("buy_and_hold", "Comprar y mantener", "#b76c28"),
    ]:
        if name in trajectories:
            values = np.r_[capital, trajectories[name].portfolio_value]
            ax.plot(dates, values, label=label, color=color, lw=1.6)
    random_curves = np.array([
        np.r_[capital, frame.portfolio_value]
        for key, frame in trajectories.items() if key.startswith("random_")
    ])
    if len(random_curves):
        ax.plot(dates, random_curves.mean(axis=0), color="#777777", label="Aleatorio media")
        ax.fill_between(dates, np.quantile(random_curves, .1, axis=0),
                        np.quantile(random_curves, .9, axis=0), color="#aaaaaa",
                        alpha=.25, label="Aleatorio percentiles 10 a 90")
    ax.set_ylabel("Valor del portafolio en unidades monetarias")
    ax.grid(alpha=.2)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "curvas_capital.png")
    plt.close(fig)
    if "rl" in trajectories:
        frame = trajectories["rl"]
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(dates, part.prices, color="#1c5369", lw=1)
        axes[0].set_ylabel("BTC/USDT")
        axes[1].step(pd.to_datetime(frame.decision_time), frame.position * 100,
                     where="post", color="#b76c28", lw=1)
        axes[1].set_ylabel("Exposición %")
        axes[1].set_yticks([0, 50, 100])
        for ax in axes:
            ax.grid(alpha=.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / "posiciones_agente.png")
        plt.close(fig)


def verify_frozen(frozen_dir: Path) -> dict:
    manifest = json.loads((frozen_dir / "PROTOCOL.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files_sha256"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"El archivo congelado cambió: {relative}")
    if manifest["random_seeds"] != RANDOM_SEEDS:
        raise ValueError("Las semillas aleatorias difieren del protocolo congelado.")
    return manifest


def claim_final(ledger: Path, record: dict, reproduce: bool) -> None:
    """Un registro exclusivo evita una segunda evaluación oficial accidental."""
    if reproduce:
        if not ledger.exists():
            raise ValueError("No existe una evaluación oficial para reproducir.")
        original = json.loads(ledger.read_text(encoding="utf-8"))
        if original.get("status") != "complete":
            raise ValueError("La evaluación oficial no está completa.")
        if original["protocol_sha256"] != record["protocol_sha256"]:
            raise ValueError("La reproducción requiere el mismo protocolo.")
        return
    with ledger.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["validation", "held_out"], default="validation")
    parser.add_argument("--model-dir", default="models/final")
    parser.add_argument("--output", default=None)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args(argv)
    if args.partition == "held_out" and args.baselines_only:
        parser.error("La prueba final evalúa conjuntamente al agente y los baselines.")
    if args.reproduce and args.partition != "held_out":
        parser.error("--reproduce se reserva para repetir el protocolo final congelado.")
    frozen_dir = resolve(args.model_dir)
    config_path = frozen_dir / "config.json"
    if args.baselines_only:
        with (PROJECT_ROOT / "configs/dqn.toml").open("rb") as f:
            config = tomllib.load(f)
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        verify_frozen(frozen_dir)
    output = resolve(args.output or f"results/{args.partition}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"La salida ya contiene archivos: {output}")
    started = datetime.now(timezone.utc).isoformat()
    record = {"status": "started", "started_utc": started,
              "partition": args.partition, "reproduction": args.reproduce}
    ledger = frozen_dir / "EVALUATION.json"
    if args.partition == "held_out":
        record["protocol_sha256"] = sha256(frozen_dir / "PROTOCOL.json")
        claim_final(ledger, record, args.reproduce)
    output.mkdir(parents=True, exist_ok=True)
    # Solo después de verificar el modelo y registrar el protocolo se simula.
    data, engineer = build_market_data()
    part = data[args.partition]
    trajectories, rows = {}, []
    policies = [("buy_and_hold", BuyAndHold(), 42)]
    policies += [(f"random_{s}", RandomAgent(s), s) for s in RANDOM_SEEDS]
    if not args.baselines_only:
        from src.agent import DQNAgent, set_global_seed
        set_global_seed(42)
        policies.insert(0, ("rl", DQNAgent.load(frozen_dir / "model.zip"), 42))
    for name, policy, seed in policies:
        frame, metrics = rollout(part, policy, config, seed)
        frame.to_csv(output / f"{name}_trajectory.csv", index=False)
        trajectories[name] = frame
        rows.append({"strategy": name, "seed": seed, **metrics})
    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(output / "metrics_by_run.csv", index=False)
    random_rows = all_metrics[all_metrics.strategy.str.startswith("random_")]
    summary = [r for r in rows if not r["strategy"].startswith("random_")]
    summary.append(aggregate_random(random_rows))
    write_json(output / "summary.json", summary)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    plot_results(trajectories, part, config["env"]["initial_capital"], output)
    record.update({
        "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "first_valuation_input": str(part.timestamps[0]),
        "last_valuation": str(part.timestamps[-1]), "n_prices": len(part.prices),
        "n_transitions": len(part.prices) - 1, "random_seeds": RANDOM_SEEDS,
        "initial_capital": config["env"]["initial_capital"],
        "transaction_cost_bps": config["env"]["transaction_cost_bps"],
        "forced_final_liquidation": False,
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "gymnasium", "stable-baselines3", "torch"]},
        "feature_names": part.feature_names,
        "scaler_mean": engineer.scaler.mean_.tolist(),
        "scaler_scale": engineer.scaler.scale_.tolist(),
        "source_commit": current_commit(),
        "summary_sha256": sha256(output / "summary.json"),
    })
    write_json(output / "execution.json", record)
    if args.partition == "held_out" and not args.reproduce:
        write_json(ledger, record)
    print(pd.DataFrame(summary)[["strategy", "total_return", "max_drawdown",
                                  "total_cost", "n_trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
