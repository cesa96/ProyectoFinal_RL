import pandas as pd
from src.features import FeatureEngineer

def test_lookahead_no_future_leakage():
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "volume": [10, 20, 30, 40, 50]
    })

    fe = FeatureEngineer()
    original = fe.compute_indicators(df.copy())

    df_future = df.copy()
    df_future.loc[4, "close"] = 999999

    modified = fe.compute_indicators(df_future)

    assert original.iloc[:3].equals(modified.iloc[:3]), \
        "Lookahead detected"
