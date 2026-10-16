import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = None

    def compute_indicators(self, df):
        out = pd.DataFrame(index=df.index)

        out["return"] = df["close"].pct_change()
        out["log_volume"] = np.log1p(df["volume"])

        sma24 = df["close"].rolling(24).mean()
        sma168 = df["close"].rolling(168).mean()

        out["dist_sma_24"] = df["close"] / sma24 - 1
        out["dist_sma_168"] = df["close"] / sma168 - 1

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        out["rsi_scaled"] = (rsi - 50) / 50

        log_ret = np.log(df["close"]).diff()
        out["volatility_24"] = log_ret.rolling(24).std()

        out["position"] = df.get("position", 0)
        out["portfolio_value"] = df.get("portfolio_value", 1)

        return out

    def fit_scaler(self, train_features):
        self.scaler = StandardScaler()
        self.scaler.fit(train_features)

    def transform(self, features):
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)
