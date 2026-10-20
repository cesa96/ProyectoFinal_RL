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

# ----------------------------------------------------------------------
# Iteración de la recompensa (Persona 5)
# ----------------------------------------------------------------------
# Diagnóstico de la v1 (docs/informe_persona4.md, sección 4.3; validación):
#   1. El agente pasa ~97 % del tiempo al 100 % y no supera a comprar y
#      mantener ni en retorno (+59,8 % frente a +79,3 %) ni en drawdown
#      (−36,2 % frente a −30,9 %).
#   2. Sus operaciones se concentran en las caídas de alta volatilidad
#      (febrero–abril de 2025): sale y entra en medio del movimiento, paga
#      costos y se pierde los rebotes ("whipsaw").
#
# Cada problema motiva una versión. Los parámetros (κ y λ) se fijaron ANTES de
# entrenar, con razonamientos sobre el periodo de entrenamiento, y no se
# ajustaron mirando la validación, para no convertir la validación en un
# segundo conjunto de entrenamiento.


class TurnoverPenaltyReward:
    """v2: retorno neto menos una penalización adicional por cambiar de posición.

    r_t = net_return_t − κ · turnover_t

    Ataca el problema 2. El costo de 10 bps ya está en net_return, pero un
    cambio de posición en medio de una caída tiene además un costo que el
    entorno no cobra: el riesgo de perder el rebote, y en la práctica el
    spread y el deslizamiento que el entorno no modela. κ actúa como un
    "costo sombra" que solo existe en la recompensa (el portafolio sigue
    pagando 10 bps), de modo que el agente solo cambie de posición cuando
    espere un movimiento claramente mayor que esa fricción.

    κ = 0,002 (20 bps por unidad negociada): con el costo real, operar le
    "cuesta" al agente 30 bps, cerca de 0,7 veces el movimiento horario
    absoluto típico de BTC en entrenamiento (0,42 %). Un valor mucho menor no
    cambiaría la conducta y uno mucho mayor congelaría al agente en la
    posición inicial (efectivo).
    """

    def __init__(self, kappa: float = 0.002):
        self.kappa = kappa

    def reset(self) -> None:
        """No guarda estado entre pasos."""

    def __call__(self, info: dict) -> float:
        return float(info["net_return"] - self.kappa * info["turnover"])


class DrawdownPenaltyReward:
    """v3: retorno neto menos una penalización cuando el drawdown se profundiza.

    r_t = net_return_t − λ · max(0, DD_(t−1) − DD_t),   DD ≤ 0

    Ataca el problema 1 (más riesgo que comprar y mantener). Solo se penaliza
    el empeoramiento del drawdown: una pérdida que lleva el portafolio más
    abajo de su máximo pesa (1 + λ) veces, mientras que las ganancias y las
    recuperaciones se premian igual que en la v1. Es una forma de aversión a
    las pérdidas; a diferencia de penalizar el nivel del drawdown en cada
    paso, no castiga al agente durante toda la recuperación por una caída que
    ya ocurrió, y la suma de las penalizaciones del episodio es comparable en
    escala con la suma de retornos.

    λ = 1: "una pérdida que profundiza la caída cuenta el doble". Es el valor
    más simple de interpretar y está en el rango de la aversión a las
    pérdidas estimada en finanzas conductuales (≈ 2).

    Necesita memoria (el drawdown del paso anterior), que se reinicia en
    reset() al comenzar cada episodio.
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self._previous_drawdown = 0.0

    def reset(self) -> None:
        self._previous_drawdown = 0.0

    def __call__(self, info: dict) -> float:
        deepening = max(0.0, self._previous_drawdown - info["drawdown"])
        self._previous_drawdown = info["drawdown"]
        return float(info["net_return"] - self.lam * deepening)


class TurnoverAndDrawdownReward:
    """v5: combina las penalizaciones de la v2 y de la v4.

    r_t = net_return_t − κ · turnover_t − λ · max(0, DD_(t−1) − DD_t)
    """

    def __init__(self, kappa: float = 0.002, lam: float = 0.02):
        self.turnover = TurnoverPenaltyReward(kappa)
        self.drawdown = DrawdownPenaltyReward(lam)

    def reset(self) -> None:
        self.drawdown.reset()

    def __call__(self, info: dict) -> float:
        # drawdown() ya incluye net_return; se resta solo la penalización por operar.
        return float(self.drawdown(info) - self.turnover.kappa * info["turnover"])


# Segunda iteración (después de observar v2 y v3 en validación):
#   - La v3 (λ = 1) llevó al agente a quedarse en efectivo el 99,5 % del tiempo.
#     La causa se puede calcular solo con ENTRENAMIENTO: comprar y mantener gana
#     en promedio 0,0074 % por hora, mientras que el drawdown se profundiza en
#     promedio 0,188 % por hora (episodios de 1.024 h). Con λ = 1 la penalización
#     esperada es ~25 veces la ganancia esperada de estar invertido; el valor de
#     λ que la iguala es ≈ 0,039.
#   - v4 usa λ = 0,02, la mitad de ese punto de equilibrio: estar invertido sigue
#     teniendo recompensa esperada positiva, pero las caídas pesan más.
#   - v5 combina la v2 (la mejor de la primera iteración) con la v4.
# λ se derivó de entrenamiento, no buscando el mejor valor en validación.
LAMBDA_BREAK_EVEN_TRAIN = 0.039

REWARDS: dict[str, Callable[[], Callable[[dict], float]]] = {
    # v1: retorno del portafolio neto de costos, r_t = (V_(t+1) - V_t) / V_t.
    "v1_net_return": NetReturnReward,
    # v2: penaliza cambiar de posición (whipsaw en alta volatilidad).
    "v2_turnover_penalty": lambda: TurnoverPenaltyReward(kappa=0.002),
    # v3: penaliza profundizar el drawdown (más riesgo que comprar y mantener).
    "v3_drawdown_penalty": lambda: DrawdownPenaltyReward(lam=1.0),
    # v4: v3 con λ escalado a partir de los datos de entrenamiento.
    "v4_drawdown_scaled": lambda: DrawdownPenaltyReward(lam=0.02),
    # v5: penalización por operar (v2) + drawdown escalado (v4).
    "v5_turnover_drawdown": lambda: TurnoverAndDrawdownReward(kappa=0.002, lam=0.02),
}


def make_reward(name: str) -> Callable[[dict], float]:
    """Crea una instancia nueva de la recompensa registrada con ese nombre."""
    if name not in REWARDS:
        raise KeyError(
            f"Recompensa desconocida: {name!r}. Disponibles: {sorted(REWARDS)}"
        )
    return REWARDS[name]()
