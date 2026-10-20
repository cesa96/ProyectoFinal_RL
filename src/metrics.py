"""Métricas financieras independientes de la recompensa de entrenamiento."""
from __future__ import annotations

import numpy as np
import pandas as pd


def financial_metrics(trajectory: pd.DataFrame, initial_capital: float) -> dict:
    """Incluye V0 en el retorno y en el máximo previo para medir drawdown."""
    if initial_capital <= 0 or trajectory.empty:
        raise ValueError("Se requiere capital positivo y una trayectoria no vacía.")
    values = np.r_[initial_capital, trajectory["portfolio_value"].to_numpy(float)]
    costs = trajectory["transaction_cost"].to_numpy(float)
    positions = trajectory["position"].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(costs).all():
        raise ValueError("Valores y costos deben ser finitos.")
    if (values <= 0).any() or (costs < 0).any():
        raise ValueError("Se requieren valores positivos y costos no negativos.")
    changes = np.abs(np.diff(np.r_[0.0, positions]))
    result = {
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / initial_capital - 1),
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
        "total_cost": float(costs.sum()),
        "n_trades": int(np.count_nonzero(changes > 1e-12)),
        "turnover": float(changes.sum()),
        "mean_exposure": float(positions.mean()),
        "n_steps": len(trajectory),
    }
    for p in (0, 50, 100):
        result[f"pct_time_{p}"] = float(np.mean(np.isclose(positions, p / 100)))
    return result


def aggregate_random(rows: pd.DataFrame) -> dict:
    """Media y desviación muestral de métricas por trayectoria, no de una curva media."""
    metrics = [c for c in rows.columns if c not in ("strategy", "seed")]
    result = {"strategy": "Aleatorio", "n_seeds": len(rows)}
    for metric in metrics:
        result[metric] = float(rows[metric].mean())
        result[f"{metric}_std"] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else 0.0
    return result
