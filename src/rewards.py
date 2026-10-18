"""Registro de funciones de recompensa.

Este archivo pertenece a la Persona 5 (diseño e iteración de la recompensa).
La Persona 4 solo dejó el registro con la versión inicial para que train.py
pueda entrenar cualquier versión por nombre:

    python -m src.train --reward v1_net_return

Para agregar una versión nueva, defina una clase invocable que reciba el
diccionario `info` del entorno y devuelva un número (opcionalmente con un
método reset()), y regístrela en REWARDS con un nombre nuevo. No modifique
versiones ya entrenadas: cree una nueva para que la comparación sea trazable.
"""

from __future__ import annotations

from typing import Callable

from src.environment import NetReturnReward

REWARDS: dict[str, Callable[[], Callable[[dict], float]]] = {
    # v1: retorno del portafolio neto de costos, r_t = (V_(t+1) - V_t) / V_t.
    "v1_net_return": NetReturnReward,
}


def make_reward(name: str) -> Callable[[dict], float]:
    """Crea una instancia nueva de la recompensa registrada con ese nombre."""
    if name not in REWARDS:
        raise KeyError(
            f"Recompensa desconocida: {name!r}. Disponibles: {sorted(REWARDS)}"
        )
    return REWARDS[name]()
