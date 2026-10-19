"""Pruebas del flujo de la Persona 5: nueva recompensa -> entrenar -> analizar."""

from __future__ import annotations

import pytest

from src import rewards
from src.analysis import compare, rollout
from src.train import main as train_main


class _ToyPenaltyReward:
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, info: dict) -> float:
        return info["net_return"] - self.penalty * info["turnover"]


@pytest.fixture
def toy_reward(monkeypatch):
    monkeypatch.setitem(rewards.REWARDS, "test_penalty",
                        lambda: _ToyPenaltyReward(penalty=0.001))
    return "test_penalty"


def test_new_reward_can_be_trained_inspected_and_compared(tmp_path, toy_reward):
    train_main(["--reward", toy_reward, "--seeds", "1", "2", "--timesteps", "6000",
                "--eval-freq", "3000", "--output-dir", str(tmp_path)])
    run_dir = tmp_path / toy_reward / "seed_1"
    assert (run_dir / "best_model.zip").exists()

    frame, stats = rollout(run_dir)
    assert len(frame) == 8760 - 1  # validación completa: 8.760 horas, 8.759 pasos
    assert set(frame["position"].unique()) <= {0.0, 0.5, 1.0}
    assert stats["reward"] == toy_reward
    assert {"q_0", "q_50", "q_100"} <= set(frame.columns)

    table = compare([toy_reward], runs_dir=tmp_path)
    assert (tmp_path / "comparison" / "reward_comparison.md").exists()
    assert table["n_seeds"].eq(2).all()


def test_analysis_refuses_held_out(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        rollout(tmp_path, partition="held_out")


def test_unknown_reward_fails_fast():
    with pytest.raises(KeyError):
        rewards.make_reward("no_existe")
