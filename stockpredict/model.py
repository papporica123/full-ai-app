"""Ridge regression with walk-forward training (no lookahead) + explanations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_features
from .graph import CausalGraph, default_graph


def _fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return mu, sd, w, ym


def _predict(m, X):
    mu, sd, w, ym = m
    return ((X - mu) / sd) @ w + ym


@dataclass
class Prediction:
    ticker: str
    horizon: int
    expected_return: float
    prob_up: float
    interval80: tuple
    drivers: list      # (feature, contribution)
    news_paths: list   # (hop, node, contribution)


class Predictor:
    def __init__(self, ticker, graph: CausalGraph | None = None, horizon=5, ridge=0.05, halflife=3.0):
        self.ticker, self.horizon, self.ridge, self.halflife = ticker, horizon, ridge, halflife
        self.graph = graph or default_graph()

    def _dataset(self, df, news):
        X, S = build_features(df, news, self.graph, self.ticker, self.halflife)
        y = np.log(df["Close"]).diff(self.horizon).shift(-self.horizon)  # forward H-day return
        return X, y, S

    def walk_forward(self, df, news, min_train=250, step=21):
        """Out-of-sample predictions: at day t train only on rows whose target ended <= t."""
        X, y, _ = self._dataset(df, news)
        ok = X.notna().all(axis=1)
        preds = pd.Series(np.nan, index=df.index)
        H, cols = self.horizon, X.columns
        for t0 in range(min_train, len(df), step):
            tr = np.where(ok & y.notna())[0]
            tr = tr[tr + H <= t0]
            if len(tr) < 100:
                continue
            m = _fit(X.iloc[tr][cols].values, y.iloc[tr].values, self.ridge)
            te = np.arange(t0, min(t0 + step, len(df)))
            te = te[ok.values[te]]
            preds.iloc[te] = _predict(m, X.iloc[te].values)
        return preds, y

    def predict(self, df, news) -> Prediction:
        X, y, S = self._dataset(df, news)
        ok = (X.notna().all(axis=1) & y.notna()).values
        tr = np.where(ok)[0]
        m = _fit(X.iloc[tr].values, y.iloc[tr].values, self.ridge)
        resid = y.iloc[tr].values - _predict(m, X.iloc[tr].values)
        sig = resid.std() + 1e-9
        x = X.iloc[[-1]].values
        mu_hat = float(_predict(m, x)[0])
        contrib = ((x[0] - m[0]) / m[1]) * m[2]
        drivers = sorted(zip(X.columns, contrib), key=lambda r: -abs(r[1]))
        return Prediction(
            self.ticker, self.horizon, mu_hat, 0.5 * (1 + math.erf(mu_hat / (sig * math.sqrt(2)))),
            (mu_hat - 1.2816 * sig, mu_hat + 1.2816 * sig),
            [(k, float(v)) for k, v in drivers[:6]],
            self.graph.explain(self.ticker, S[-1]),
        )
