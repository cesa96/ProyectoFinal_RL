"""Agente DQN.

Responsabilidades de este módulo:

* Leer la configuración del agente (configs/dqn.toml).
* Fijar todas las semillas (Python, NumPy, PyTorch y entorno).
* Construir el modelo DQN de Stable-Baselines3 con los hiperparámetros del
  archivo de configuración.
* Ofrecer la interfaz `agent.act(obs)` del enunciado para que la evaluación y
  los baselines usen al agente igual que a cualquier otra estrategia.
* Recorrer un episodio completo con una política y resumirlo (lo usa train.py
  para evaluar en validación durante el entrenamiento).

Por qué DQN: el espacio de acciones es discreto y pequeño (0 %, 50 %, 100 %),
que es exactamente el caso para el que fue diseñado DQN (aprende un valor Q por
acción). Además es off-policy y reutiliza las transiciones del replay buffer,
lo que aprovecha mejor una serie histórica de tamaño fijo.
"""

from __future__ import annotations

import os
import random
import tomllib
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.environment import TradingEnvironment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DQN_CONFIG = PROJECT_ROOT / "configs" / "dqn.toml"

_ACTIVATIONS = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}


# ----------------------------------------------------------------------
# Configuración y semillas
# ----------------------------------------------------------------------
def load_dqn_config(path: str | Path = DEFAULT_DQN_CONFIG) -> dict:
    """Lee configs/dqn.toml."""
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def set_global_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch.

    La semilla del entorno se fija aparte con env.reset(seed=...) y la de la
    exploración del agente la fija Stable-Baselines3 con el argumento `seed`.
    Se limita PyTorch a un hilo porque en CPU el paralelismo puede cambiar el
    orden de las sumas y, con él, los últimos decimales de los pesos.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------
# Construcción y carga del modelo
# ----------------------------------------------------------------------
def build_dqn(env, config: dict, seed: int, verbose: int = 0):
    """Crea el modelo DQN con los hiperparámetros de la configuración."""
    from stable_baselines3 import DQN

    agent_cfg = config["agent"]
    exploration = config["exploration"]
    if agent_cfg.get("algorithm", "DQN") != "DQN":
        raise ValueError("build_dqn solo construye agentes DQN.")

    policy_kwargs = {
        "net_arch": list(agent_cfg["net_arch"]),
        "activation_fn": _ACTIVATIONS[agent_cfg.get("activation", "relu")],
    }
    set_global_seed(seed)
    return DQN(
        policy=agent_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=agent_cfg["learning_rate"],
        buffer_size=agent_cfg["buffer_size"],
        learning_starts=agent_cfg["learning_starts"],
        batch_size=agent_cfg["batch_size"],
        gamma=agent_cfg["gamma"],
        train_freq=agent_cfg["train_freq"],
        gradient_steps=agent_cfg["gradient_steps"],
        target_update_interval=agent_cfg["target_update_interval"],
        max_grad_norm=agent_cfg.get("max_grad_norm", 10.0),
        exploration_fraction=exploration["exploration_fraction"],
        exploration_initial_eps=exploration["exploration_initial_eps"],
        exploration_final_eps=exploration["exploration_final_eps"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )


class DQNAgent:
    """Envoltura con la interfaz del enunciado: action = agent.act(obs).

    Siempre actúa de forma determinista (acción con mayor valor Q, sin
    exploración), que es como se evalúa al agente.
    """

    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        from stable_baselines3 import DQN

        return cls(DQN.load(str(path), device="cpu"))

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Valores Q de las tres acciones (útil para analizar decisiones)."""
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            return self.model.q_net(obs_tensor).cpu().numpy().ravel()


# ----------------------------------------------------------------------
# Recorrido de un episodio
# ----------------------------------------------------------------------
def run_episode(
    env: TradingEnvironment,
    act: Callable[[np.ndarray], int],
    seed: int | None = None,
) -> dict:
    """Recorre un episodio completo y devuelve un resumen.

    `env` debe haberse creado con record_history=True. Las métricas aquí son
    las mínimas para seleccionar checkpoints en validación; la evaluación final
    y su tabla de resultados se calculan con src/metrics.py.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    info: dict = {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(act(obs))
        total_reward += reward
        done = terminated or truncated
    history = env.get_history()
    return summarize_history(history, total_reward, env.fractions)


def summarize_history(history: dict, total_reward: float, fractions) -> dict:
    values = np.concatenate([[history["initial_capital"]], history["portfolio_value"]])
    peaks = np.maximum.accumulate(values)
    max_drawdown = float((values / peaks - 1.0).min())
    total_return = float(values[-1] / values[0] - 1.0)
    positions = history["position"]
    changes = np.abs(np.diff(np.concatenate([[0.0], positions]))) > 0
    summary = {
        "total_reward": float(total_reward),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "calmar": total_return / abs(max_drawdown) if max_drawdown < 0 else float("inf"),
        "total_cost": float(history["transaction_cost"].sum()),
        "n_trades": int(changes.sum()),
        "final_value": float(values[-1]),
    }
    for fraction in fractions:
        summary[f"pct_time_{int(round(fraction * 100))}"] = float(
            np.mean(np.isclose(positions, fraction))
        )
    return summary
