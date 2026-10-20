import pandas as pd
from src.features import FeatureEngineer

def test_lookahead_no_future_leakage():
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "close": 100 * np.exp(np.cumsum(rng.normal(0, .01, 420))),
        "volume": rng.uniform(10, 100, 420)
    })

    fe = FeatureEngineer()
    original = fe.compute_indicators(df.copy())

    df_future = df.copy()
    df_future.loc[301:, "close"] *= 100
    df_future.loc[301:, "volume"] *= 50

    modified = fe.compute_indicators(df_future)

    assert original.loc[200:300].notna().all().all()
    assert original.iloc[:301].equals(modified.iloc[:301]), \
        "Lookahead detected"
