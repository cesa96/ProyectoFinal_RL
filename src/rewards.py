"""Registro de funciones de recompensa.

Este archivo pertenece a la Persona 5 (diseño e iteración de la recompensa).
La Persona 4 solo dejó el registro con la versión inicial para que train.py
pueda entrenar cualquier versión por nombre:

    python -m src.train --reward v1_net_return

Para agregar una versión nueva, defina una clase invocable que reciba el
diccionario `info` del entorno y devuelva un número (opcionalmente con un
método reset()), y regístrela en REWARDS con un nombre nuevo. No modifique
versiones ya entrenadas: cree una nueva para que la comparación sea trazable.

Claves disponibles en `info` en cada paso (las calcula el entorno):
    net_return, log_return       retorno del portafolio en el paso, neto de costos
    gross_return                 retorno del activo de close_t a close_(t+1)
    asset_return_exposed         posición * gross_return
    position, previous_position  fracción invertida ahora y antes (0, 0.5, 1)
    turnover                     |posición nueva - posición anterior|
    transaction_cost             costo pagado en dinero en este paso
    cost_fraction                costo / valor del portafolio antes del paso
    portfolio_value, previous_value, peak_value
    drawdown                     valor / máximo alcanzado - 1 (<= 0)
    n_trades, total_cost         acumulados del episodio
    step, index, price, next_price

Ejemplo de una versión con parámetro (ilustrativo, no está registrado):

    class TurnoverPenaltyReward:
        def __init__(self, penalty: float = 0.001):
            self.penalty = penalty

        def __call__(self, info: dict) -> float:
            return info["net_return"] - self.penalty * info["turnover"]

    REWARDS["v2_turnover_penalty"] = lambda: TurnoverPenaltyReward(penalty=0.001)

Después:
    python -m src.train --reward v2_turnover_penalty
    python -m src.analysis inspect runs/v2_turnover_penalty/seed_42
    python -m src.analysis compare v1_net_return v2_turnover_penalty

Si la recompensa necesita memoria entre pasos (por ejemplo, una media móvil de
retornos para penalizar volatilidad), guárdela en atributos de la clase y
reiníciela en reset(); el entorno llama reset() al comenzar cada episodio.
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
