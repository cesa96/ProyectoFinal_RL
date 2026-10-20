"""Estrategias de referencia con la misma interfaz que el agente."""
from __future__ import annotations

import numpy as np


class BuyAndHold:
    """Elige 100 % en todos los pasos; el entorno cobra la entrada inicial."""

    def act(self, observation) -> int:
        return 2


class RandomAgent:
    """Acciones uniformes independientes; una semilla por trayectoria."""

    def __init__(self, seed: int, n_actions: int = 3):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, observation) -> int:
        return int(self.rng.integers(self.n_actions))


def main() -> None:
    """Ejecuta baselines en validación; la evaluación final usa src.evaluate."""
    from src.evaluate import main as evaluate_main
    evaluate_main(["--partition", "validation", "--baselines-only",
                   "--output", "results/validation_baselines"])


if __name__ == "__main__":
    main()
