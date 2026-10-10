"""Feature engineering: technicals + multi-hop news impact."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .news import Headline, extract_shocks, sentiment

HOP_COLS = ["impact_direct", "impact_hop1", "impact_hop2", "impact_hop3"]


def shock_state(news: list[Headline], dates: pd.DatetimeIndex, graph, halflife: float = 3.0,
                ticker: str | None = None):
    """Exponentially-decayed node shock vector per date, shape (T, n_nodes).
    Propagation is linear so we smooth first and propagate once. Headlines are
    assigned to the first trading day >= their date (no lookahead)."""
    lam = 0.5 ** (1.0 / halflife)
    raw = np.zeros((len(dates), len(graph.nodes)))
    for h in news:
        if h.ticker and h.ticker != ticker:
            continue
        i = dates.searchsorted(h.ts.normalize(), side="left")
        if i < len(dates):
            raw[i] += graph.shock_vector(extract_shocks(h.text))
    out = np.zeros_like(raw)
    acc = np.zeros(raw.shape[1])
    for t in range(len(dates)):
        acc = lam * acc + raw[t]
        out[t] = acc
    return out


def sentiment_series(news, dates, ticker, halflife=3.0):
    lam = 0.5 ** (1.0 / halflife)
    raw = np.zeros(len(dates))
    for h in news:
        if h.ticker == ticker:
            i = dates.searchsorted(h.ts.normalize(), side="left")
            if i < len(dates):
                raw[i] += sentiment(h.text)
    out, acc = np.zeros_like(raw), 0.0
    for t in range(len(raw)):
        acc = lam * acc + raw[t]
        out[t] = acc
    return out


def technicals(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    ret = np.log(c).diff()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f = pd.DataFrame(index=df.index)
    f["ret_1"], f["ret_5"], f["ret_20"] = ret, ret.rolling(5).sum(), ret.rolling(20).sum()
    f["vol_20"] = ret.rolling(20).std()
    f["rsi"] = (100 - 100 / (1 + gain / (loss + 1e-12)) - 50) / 50
    f["ma_gap"] = c / c.rolling(50).mean() - 1
    f["vol_surge"] = np.log(df["Volume"] / df["Volume"].rolling(20).mean())
    return f


def build_features(df: pd.DataFrame, news: list[Headline], graph, ticker: str, halflife: float = 3.0):
    f = technicals(df)
    S = shock_state(news, df.index, graph, halflife, ticker)
    imp = graph.impact_by_hop(ticker, S)
    for k, col in enumerate(HOP_COLS):
        f[col] = imp[:, k]
    f["sentiment"] = sentiment_series(news, df.index, ticker, halflife)
    return f, S
