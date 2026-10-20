import json

import numpy as np
import pandas as pd
import pytest

from src.baselines import BuyAndHold, RandomAgent
from src.env_factory import MarketData
from src.evaluate import claim_final, rollout
from src.metrics import aggregate_random, financial_metrics


def sample_part():
    return MarketData(np.array([100., 110., 99., 120.]), np.zeros((4, 2)),
                      list(pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")),
                      ["a", "b"])


CONFIG = {"env": {"initial_capital": 1000., "transaction_cost_bps": 10.},
          "training": {"reward": "v1_net_return"}}


def test_buy_hold_includes_entry_cost_and_correct_timestamp():
    part = sample_part()
    trajectory, stats = rollout(part, BuyAndHold(), CONFIG)
    assert stats["final_value"] == pytest.approx(1000 * .999 * 120 / 100)
    assert stats["total_cost"] == 1
    assert stats["n_trades"] == 1
    assert len(trajectory) == 3
    assert trajectory.decision_time.iloc[0] == part.timestamps[0]
    assert trajectory.valuation_time.iloc[0] == part.timestamps[1]


def test_drawdown_includes_initial_loss():
    frame = pd.DataFrame({"portfolio_value": [900., 950.],
                          "transaction_cost": [100., 0.], "position": [1., 1.]})
    stats = financial_metrics(frame, 1000)
    assert stats["max_drawdown"] == pytest.approx(-.1)
    assert stats["total_return"] == pytest.approx(-.05)


def test_cash_preserves_capital():
    class Cash:
        def act(self, obs):
            return 0
    _, stats = rollout(sample_part(), Cash(), CONFIG)
    assert stats["final_value"] == 1000
    assert stats["total_cost"] == 0
    assert stats["max_drawdown"] == 0


def test_random_reproducibility_and_aggregation():
    first, _ = rollout(sample_part(), RandomAgent(42), CONFIG)
    second, _ = rollout(sample_part(), RandomAgent(42), CONFIG)
    pd.testing.assert_frame_equal(first, second)
    result = aggregate_random(pd.DataFrame({"strategy": ["a", "b"], "seed": [1, 2],
                                           "max_drawdown": [-.1, -.3]}))
    assert result["max_drawdown"] == pytest.approx(-.2)
    assert result["max_drawdown_std"] == pytest.approx(np.std([-.1, -.3], ddof=1))


def test_official_evaluation_cannot_be_repeated(tmp_path):
    ledger = tmp_path / "EVALUATION.json"
    record = {"status": "started", "protocol_sha256": "abc"}
    claim_final(ledger, record, False)
    with pytest.raises(FileExistsError):
        claim_final(ledger, record, False)
    with pytest.raises(ValueError):
        claim_final(ledger, record, True)
    ledger.write_text(json.dumps({**record, "status": "complete"}))
    claim_final(ledger, record, True)
    with pytest.raises(ValueError):
        claim_final(ledger, {**record, "protocol_sha256": "other"}, True)
