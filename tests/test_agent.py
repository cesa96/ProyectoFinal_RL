"""Pruebas del agente y del protocolo de entrenamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent import DQNAgent, build_dqn, load_dqn_config, run_episode, summarize_history
from src.data import load_split_config
from src.environment import TradingEnvironment

REQUIRED = {
    "agent": ["net_arch", "learning_rate", "batch_size", "gamma", "buffer_size",
              "learning_starts", "train_freq", "target_update_interval"],
    "exploration": ["exploration_initial_eps", "exploration_final_eps",
                    "exploration_fraction"],
    "training": ["total_timesteps", "seeds", "reward"],
    "selection": ["eval_freq", "metric", "patience"],
}


def _toy_env(n: int = 400, seed: int = 0) -> TradingEnvironment:
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    features = rng.normal(size=(n, 3))
    return TradingEnvironment(prices, features, episode_length=100, random_start=True)


def _small_config() -> dict:
    config = load_dqn_config()
    config["agent"].update(learning_starts=50, buffer_size=1_000, batch_size=16,
                           target_update_interval=100)
    return config


def test_config_documents_all_hyperparameters():
    config = load_dqn_config()
    for section, keys in REQUIRED.items():
        for key in keys:
            assert key in config[section], f"falta {section}.{key} en configs/dqn.toml"
    assert len(config["training"]["seeds"]) >= 3


def test_same_seed_gives_identical_weights():
    config = _small_config()
    params = []
    for _ in range(2):
        model = build_dqn(_toy_env(), config, seed=7)
        model.learn(400)
        params.append([p.detach().clone() for p in model.q_net.parameters()])
    for a, b in zip(*params):
        assert np.array_equal(a.numpy(), b.numpy())


def test_different_seeds_give_different_weights():
    config = _small_config()
    first = build_dqn(_toy_env(), config, seed=1)
    second = build_dqn(_toy_env(), config, seed=2)
    a = next(first.q_net.parameters()).detach().numpy()
    b = next(second.q_net.parameters()).detach().numpy()
    assert not np.array_equal(a, b)


def test_agent_act_returns_valid_action():
    env = _toy_env()
    agent = DQNAgent(build_dqn(env, _small_config(), seed=0))
    obs, _ = env.reset(seed=0)
    assert env.action_space.contains(agent.act(obs))
    assert agent.q_values(obs).shape == (3,)


def test_summary_matches_hand_calculation():
    history = {
        "initial_capital": 100.0,
        "portfolio_value": np.array([110.0, 99.0, 120.0]),
        "position": np.array([1.0, 1.0, 0.5]),
        "transaction_cost": np.array([0.1, 0.0, 0.05]),
    }
    summary = summarize_history(history, total_reward=0.0, fractions=[0.0, 0.5, 1.0])
    assert summary["total_return"] == pytest.approx(0.20)
    assert summary["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert summary["n_trades"] == 2
    assert summary["total_cost"] == pytest.approx(0.15)
    assert summary["pct_time_100"] == pytest.approx(2 / 3)


def test_run_episode_buy_and_hold_on_toy_env():
    rng = np.random.default_rng(3)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
    env = TradingEnvironment(prices, rng.normal(size=(50, 2)), record_history=True,
                             transaction_cost_bps=0.0)
    summary = run_episode(env, lambda obs: 2)
    assert summary["total_return"] == pytest.approx(prices[-1] / prices[0] - 1)


def test_training_never_builds_held_out_environment():
    from src.train import make_train_and_validation_envs

    config = load_dqn_config()
    train_env, validation_env = make_train_and_validation_envs(config)
    # El último timestamp que ve el entrenamiento (validación) es anterior al
    # inicio del held-out.
    held_out_start = pd.Timestamp(load_split_config()["validation_end_exclusive"])
    assert pd.Timestamp(validation_env.timestamps[-1]) < held_out_start
    assert pd.Timestamp(train_env.timestamps[-1]) < pd.Timestamp(validation_env.timestamps[0])


def test_set_override_changes_only_requested_value():
    from src.train import apply_overrides, parse_args

    base = load_dqn_config()
    config = apply_overrides(base, parse_args(["--set", "agent.gamma=0.95"]))
    assert config["agent"]["gamma"] == 0.95
    assert config["agent"]["learning_rate"] == base["agent"]["learning_rate"]
    assert base["agent"]["gamma"] == 0.99  # el archivo base no se modifica
    with pytest.raises(ValueError):
        apply_overrides(base, parse_args(["--set", "agent.no_existe=1"]))
