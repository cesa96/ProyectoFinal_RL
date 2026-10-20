"""Pruebas de las funciones de recompensa."""

import numpy as np
import pytest

from src.environment import TradingEnvironment
from src.rewards import REWARDS, DrawdownPenaltyReward, TurnoverPenaltyReward, make_reward


def _env(prices, reward_fn):
    prices = np.asarray(prices, dtype=float)
    return TradingEnvironment(prices=prices, features=np.zeros((len(prices), 1)),
                              reward_fn=reward_fn)


def test_all_registered_rewards_can_be_built():
    for name in REWARDS:
        reward = make_reward(name)
        assert callable(reward)


def test_turnover_penalty_equals_v1_when_position_does_not_change():
    info = {"net_return": 0.01, "turnover": 0.0}
    assert TurnoverPenaltyReward(kappa=0.002)(info) == pytest.approx(0.01)


def test_turnover_penalty_charges_kappa_per_unit_traded():
    reward = TurnoverPenaltyReward(kappa=0.002)
    assert reward({"net_return": 0.0, "turnover": 1.0}) == pytest.approx(-0.002)
    assert reward({"net_return": 0.0, "turnover": 0.5}) == pytest.approx(-0.001)


def test_turnover_penalty_does_not_change_portfolio_value():
    """La penalización solo vive en la recompensa: el entorno sigue cobrando 10 bps."""
    prices = [100, 101, 99, 102, 100]
    actions = [2, 0, 2, 0]
    values = {}
    for name in ("v1_net_return", "v2_turnover_penalty"):
        env = _env(prices, make_reward(name))
        env.reset()
        for a in actions:
            _, _, _, _, info = env.step(a)
        values[name] = info["portfolio_value"]
    assert values["v1_net_return"] == pytest.approx(values["v2_turnover_penalty"])


def test_drawdown_penalty_only_when_drawdown_deepens():
    reward = DrawdownPenaltyReward(lam=1.0)
    reward.reset()
    # Ganancia en un máximo: sin penalización.
    assert reward({"net_return": 0.02, "drawdown": 0.0}) == pytest.approx(0.02)
    # Cae 5 % desde el máximo: la pérdida cuenta doble.
    assert reward({"net_return": -0.05, "drawdown": -0.05}) == pytest.approx(-0.10)
    # Se recupera parcialmente: igual que la v1.
    assert reward({"net_return": 0.03, "drawdown": -0.02}) == pytest.approx(0.03)


def test_drawdown_penalty_resets_between_episodes():
    reward = DrawdownPenaltyReward(lam=1.0)
    reward({"net_return": -0.1, "drawdown": -0.1})
    reward.reset()
    # Tras el reset el drawdown de referencia es 0: una caída de 1 % cuenta doble
    # respecto a 0, no respecto a −10 %.
    assert reward({"net_return": -0.01, "drawdown": -0.01}) == pytest.approx(-0.02)


def test_combined_reward_adds_both_penalties():
    from src.rewards import TurnoverAndDrawdownReward

    reward = TurnoverAndDrawdownReward(kappa=0.002, lam=0.5)
    reward.reset()
    value = reward({"net_return": -0.04, "drawdown": -0.04, "turnover": 1.0})
    assert value == pytest.approx(-0.04 - 0.5 * 0.04 - 0.002)
    reward.reset()
    assert reward({"net_return": 0.01, "drawdown": 0.0, "turnover": 0.0}) == pytest.approx(0.01)


def test_drawdown_penalty_in_environment_matches_formula():
    prices = [100, 110, 99, 105, 90]
    env = _env(prices, DrawdownPenaltyReward(lam=1.0))
    env.reset()
    previous_dd = 0.0
    for _ in range(len(prices) - 1):
        _, reward, _, _, info = env.step(2)
        expected = info["net_return"] - max(0.0, previous_dd - info["drawdown"])
        assert reward == pytest.approx(expected)
        previous_dd = info["drawdown"]
