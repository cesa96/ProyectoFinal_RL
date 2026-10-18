"""Entorno de simulación de inversión
El entorno representa a un inversionista que, en cada hora, decide qué fracción
de su capital mantiene invertida en un único activo (BTC/USDT). Por esta razón,
la fracción siempre se encuentra entre 0 y 1: no se permiten posiciones en
corto ni apalancamiento.

Cómo se construyó el entorno
----------------------------
El diseño se desarrolló alrededor de tres reglas: que el valor del
portafolio se calcule correctamente, que operar tenga un costo y que el agente
nunca reciba información del futuro. A continuación se describen las decisiones
tomadas y la razón de cada una.

1. Acciones. Se utiliza un espacio discreto de tres acciones (0 %, 50 % y 100 %
   del capital). Este espacio es fácil de interpretar económicamente y es
   compatible con DQN. Como consecuencia, el agente no puede ir en corto, no
   puede usar apalancamiento y tampoco puede invertir una fracción como 25 %.

2. Observación. La observación está formada por las variables de mercado ya
   escaladas y por la posición actual del agente. La posición se agrega dentro
   del entorno y no pasa por el escalador. Las columnas position y
   portfolio_value describen al agente y no al mercado, y en compute_indicators
   se rellenan con valores constantes (0 y 1) cuando el DataFrame no las trae. Por
   esta razón, escalarlas con la media y la desviación del entrenamiento carece de
   sentido, ya que la desviación es cero. Por la misma razón, portfolio_value no
   se incluye como variable de observación.

3. Línea de tiempo de un paso:
   a. En el instante t el agente recibe obs_t, construida únicamente con la
      información disponible hasta el cierre de la vela t.
   b. El agente elige la acción a_t, que corresponde a la fracción objetivo del
      capital que desea tener en el activo durante el siguiente intervalo.
   c. Si la fracción objetivo es distinta de la posición actual, se opera al
      precio de cierre close_t y se paga el costo de transacción.
   d. La posición elegida queda expuesta al movimiento del precio de close_t a
      close_(t+1). Este es el único momento en el que el entorno consulta el
      precio futuro, y lo hace solamente para liquidar el resultado del paso,
      nunca para construir la observación.
   e. Se calcula la recompensa con el valor del portafolio antes y después del
      paso, y se entrega obs_(t+1).

4. Cálculo del valor del portafolio:
       turnover      = |posición_nueva - posición_anterior|
       costo         = costo_por_unidad * turnover * V_t
       V_t_neto      = V_t - costo
       retorno_bruto = close_(t+1) / close_t - 1
       V_(t+1)       = V_t_neto * (1 + posición_nueva * retorno_bruto)
   El costo se cobra sobre el valor negociado, es decir, sobre la parte del
   capital que cambia de lado, y se descuenta del portafolio antes de aplicar el
   movimiento del precio. De esta manera, una estrategia que cambia de posición
   sin obtener ganancia solamente pierde capital por costos.

5. Final del episodio. Llegar al final de los datos se reporta como truncated y
   no como terminated, porque se trata de un corte por tiempo y no de un final
   natural del problema. Solamente se reporta terminated si el portafolio llega
   a cero.

6. Recompensa. Se implementó como un componente intercambiable: puede ser
   cualquier objeto invocable que reciba el diccionario info del paso y devuelva
   un número. Si además tiene un método reset(), el entorno lo llama al iniciar
   cada episodio. De esta manera, el diseño de la recompensa se puede iterar
   (retorno neto, retorno logarítmico, penalizaciones por caída o por operar) sin
   modificar este archivo.
"""

from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Fracciones del capital asociadas a las acciones 0, 1 y 2, que equivalen a
# estar en efectivo, invertir la mitad del capital e invertirlo todo.
ACTION_FRACTIONS: tuple[float, ...] = (0.0, 0.5, 1.0)

# Un basis point equivale a 0,01 %, por lo tanto diez basis points son 0,10 %.
BPS = 1e-4


class NetReturnReward:
    """Versión inicial de la recompensa: retorno del portafolio neto de costos.

    r_t = (V_(t+1) - V_t) / V_t

    Dado que V_(t+1) ya tiene descontado el costo de transacción, la recompensa
    incorpora el costo de operar sin necesidad de agregar un término adicional.
    """

    def reset(self) -> None:
        """Esta recompensa no guarda estado entre episodios."""

    def __call__(self, info: dict) -> float:
        return float(info["net_return"])


class TradingEnvironment(gym.Env):
    """Entorno de inversión sobre una serie de precios de un solo activo.

    Parámetros
    ----------
    prices:
        Precios de cierre, un valor por vela, en orden cronológico.
    features:
        Matriz (n_velas, n_features) con las variables de mercado ya escaladas.
        La fila i solo puede contener información disponible hasta la vela i. No
        debe incluir la posición, porque el entorno la agrega por su cuenta.
    reward_fn:
        Objeto invocable que recibe info y devuelve la recompensa. Por defecto
        se utiliza NetReturnReward.
    initial_capital:
        Capital inicial expresado en dinero.
    transaction_cost_bps:
        Costo de transacción en basis points sobre el valor negociado.
    action_fractions:
        Fracción del capital asociada a cada acción discreta.
    episode_length:
        Número de pasos por episodio. Si es None se recorre toda la serie.
    random_start:
        Si es True, cada episodio comienza en una vela al azar, usando la
        semilla del entorno. Esta opción está pensada para el entrenamiento; para
        evaluar se debe dejar en False y recorrer la partición completa.
    timestamps:
        Marcas de tiempo opcionales, utilizadas solamente para el historial.
    record_history:
        Si es True, guarda el recorrido del episodio para graficarlo
        posteriormente con get_history().
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: Sequence[float],
        features: np.ndarray,
        reward_fn: Callable[[dict], float] | None = None,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,
        action_fractions: Sequence[float] = ACTION_FRACTIONS,
        episode_length: int | None = None,
        random_start: bool = False,
        timestamps: Sequence | None = None,
        record_history: bool = False,
    ) -> None:
        super().__init__()

        prices = np.asarray(prices, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)
        fractions = np.asarray(action_fractions, dtype=np.float64)

        if prices.ndim != 1 or len(prices) < 2:
            raise ValueError("prices debe ser un vector con al menos dos precios.")
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("prices debe contener únicamente valores finitos y positivos.")
        if features.ndim != 2 or len(features) != len(prices):
            raise ValueError(
                "features debe ser una matriz con una fila por cada precio."
            )
        if not np.all(np.isfinite(features)):
            raise ValueError(
                "features contiene NaN o valores infinitos. Por esta razón, se "
                "debe eliminar el periodo de calentamiento de los indicadores "
                "antes de crear el entorno."
            )
        if fractions.ndim != 1 or len(fractions) < 2:
            raise ValueError("action_fractions debe tener al menos dos valores.")
        if np.any(fractions < 0) or np.any(fractions > 1):
            raise ValueError(
                "Las fracciones deben estar entre 0 y 1, dado que el entorno no "
                "permite posiciones en corto ni apalancamiento."
            )
        if initial_capital <= 0:
            raise ValueError("initial_capital debe ser positivo.")
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps no puede ser negativo.")
        if timestamps is not None and len(timestamps) != len(prices):
            raise ValueError("timestamps debe tener la misma longitud que prices.")

        max_steps = len(prices) - 1
        if episode_length is None:
            episode_length = max_steps
        if not 1 <= episode_length <= max_steps:
            raise ValueError(
                f"episode_length debe estar entre 1 y {max_steps}."
            )

        self.prices = prices
        self.features = features
        self.fractions = fractions
        self.initial_capital = float(initial_capital)
        self.cost_rate = float(transaction_cost_bps) * BPS
        self.episode_length = int(episode_length)
        self.random_start = bool(random_start)
        self.timestamps = None if timestamps is None else list(timestamps)
        self.record_history = bool(record_history)
        self.reward_fn = reward_fn if reward_fn is not None else NetReturnReward()

        self.n_market_features = features.shape[1]
        self.action_space = spaces.Discrete(len(fractions))
        # La observación se compone del vector de mercado y de la posición actual
        # del agente.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_market_features + 1,),
            dtype=np.float32,
        )

        self._start = 0
        self._t = 0
        self._end = 0
        self._position = 0.0
        self._value = self.initial_capital
        self._peak = self.initial_capital
        self._n_trades = 0
        self._total_cost = 0.0
        self._done = True
        self._history: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Construcción de la observación
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        """Observación en el instante t: fila t de las features y la posición.

        Únicamente se lee la fila self._t, de tal manera que nunca se accede a
        información de filas posteriores.
        """
        market = self.features[self._t]
        return np.append(market, self._position).astype(np.float32)

    # ------------------------------------------------------------------
    # API de Gymnasium
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Inicia un episodio con todo el capital en efectivo (posición 0)."""
        super().reset(seed=seed)

        max_start = len(self.prices) - 1 - self.episode_length
        if options is not None and "start" in options:
            start = int(options["start"])
            if not 0 <= start <= max_start:
                raise ValueError(f"start debe estar entre 0 y {max_start}.")
        elif self.random_start and max_start > 0:
            start = int(self.np_random.integers(0, max_start + 1))
        else:
            start = 0

        self._start = start
        self._t = start
        self._end = start + self.episode_length
        self._position = 0.0
        self._value = self.initial_capital
        self._peak = self.initial_capital
        self._n_trades = 0
        self._total_cost = 0.0
        self._done = False

        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()

        self._history = {
            "step": [],
            "timestamp": [],
            "price": [],
            "position": [],
            "portfolio_value": [],
            "transaction_cost": [],
            "reward": [],
        }

        info = {
            "step": 0,
            "index": self._t,
            "portfolio_value": self._value,
            "position": self._position,
        }
        return self._observation(), info

    def step(self, action):
        """Ejecuta la acción elegida en t y avanza hasta t+1."""
        if self._done:
            raise RuntimeError(
                "El episodio ya terminó, por lo tanto se debe llamar a reset() "
                "antes de volver a usar step()."
            )
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}.")

        t = self._t
        price_now = self.prices[t]
        price_next = self.prices[t + 1]  # se usa solo para liquidar el resultado

        value_before = self._value
        previous_position = self._position
        new_position = float(self.fractions[int(action)])

        # 1. Costo de transacción sobre el valor negociado, que se paga en close_t.
        turnover = abs(new_position - previous_position)
        cost = self.cost_rate * turnover * value_before
        value_after_cost = value_before - cost

        # 2. La posición elegida queda expuesta al movimiento de close_t a close_(t+1).
        gross_return = price_next / price_now - 1.0
        value_after = value_after_cost * (1.0 + new_position * gross_return)

        # 3. Se actualiza el estado interno del entorno.
        self._value = value_after
        self._position = new_position
        self._peak = max(self._peak, value_after)
        self._total_cost += cost
        if turnover > 0:
            self._n_trades += 1
        self._t = t + 1

        net_return = value_after / value_before - 1.0
        drawdown = value_after / self._peak - 1.0

        info = {
            "step": self._t - self._start,
            "index": self._t,
            "price": price_now,
            "next_price": price_next,
            "previous_position": previous_position,
            "position": new_position,
            "turnover": turnover,
            "transaction_cost": cost,
            "cost_fraction": cost / value_before,
            "gross_return": gross_return,
            "asset_return_exposed": new_position * gross_return,
            "previous_value": value_before,
            "portfolio_value": value_after,
            "net_return": net_return,
            "log_return": float(np.log(value_after / value_before))
            if value_after > 0
            else float("-inf"),
            "peak_value": self._peak,
            "drawdown": drawdown,
            "n_trades": self._n_trades,
            "total_cost": self._total_cost,
        }

        reward = float(self.reward_fn(info))

        # Sin apalancamiento, el portafolio solo llega a cero si el activo pierde
        # todo su valor.
        terminated = bool(value_after <= 0)
        # Llegar al final de los datos representa un corte por tiempo y no un final
        # natural del episodio, por esta razón se usa truncated.
        truncated = bool(self._t >= self._end)
        self._done = terminated or truncated

        if self.record_history:
            h = self._history
            h["step"].append(info["step"])
            h["timestamp"].append(
                None if self.timestamps is None else self.timestamps[t]
            )
            h["price"].append(price_now)
            h["position"].append(new_position)
            h["portfolio_value"].append(value_after)
            h["transaction_cost"].append(cost)
            h["reward"].append(reward)

        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Utilidades para la evaluación
    # ------------------------------------------------------------------
    def get_history(self) -> dict[str, np.ndarray]:
        """Devuelve el recorrido del último episodio.

        Para poder usarlo, el entorno debe haberse creado con record_history=True.
        """
        if not self.record_history:
            raise RuntimeError("El entorno debe crearse con record_history=True.")
        history = {k: np.asarray(v) for k, v in self._history.items()}
        history["initial_capital"] = self.initial_capital
        return history
