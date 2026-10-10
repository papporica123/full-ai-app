"""Price providers: yfinance, CSV, or synthetic (offline demo)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .news import Headline

# headline templates that extract_shocks maps back to (node, sign)
TEMPLATES = {
    ("oil", +1): "Oil prices surge as supply fears grow",
    ("oil", -1): "Crude prices plunge on demand worries",
    ("middle_east", +1): "Missile attack escalates tension in Israel and Iran",
    ("middle_east", -1): "Ceasefire talks calm Gaza conflict",
    ("russia", +1): "New sanctions as Russia Ukraine war escalates",
    ("taiwan", +1): "Tension rises around Taiwan as military drills begin",
    ("rates", +1): "Fed signals rate hike, hawkish tone",
    ("rates", -1): "Fed hints at rate cut, dovish pivot",
    ("inflation", +1): "Inflation hotter than expected, CPI rises",
    ("inflation", -1): "Inflation cools, CPI falls",
    ("china_demand", -1): "China slowdown deepens as property crisis weighs",
    ("china_demand", +1): "China stimulus sparks recovery hopes",
    ("trade_war", +1): "New tariffs announced, trade war fears",
    ("banks_stress", +1): "Regional bank failure sparks credit crunch fears",
}


def load_yfinance(ticker: str, years: int = 5) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def synthetic(ticker: str, graph, days: int = 1000, seed: int = 0, signal: float = 0.004):
    """Synthetic prices + news where returns genuinely respond (with lag and noise)
    to multi-hop impacts. Validates the plumbing only - it is NOT evidence of real
    predictive power."""
    from .features import shock_state

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    keys = list(TEMPLATES)
    news = []
    for d in dates:
        for _ in range(rng.poisson(0.6)):
            node, sign = keys[rng.integers(len(keys))]
            news.append(Headline(d, TEMPLATES[(node, sign)]))
    S = shock_state(news, dates, graph, halflife=3)
    g = graph.impact_by_hop(ticker, S)
    lag_signal = np.roll(g @ np.array([1.0, 0.8, 0.5, 0.3]), 1)  # news leads returns by a day
    lag_signal[0] = 0
    r = signal * lag_signal + rng.normal(0.0003, 0.013, days)
    close = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"Open": close * (1 + rng.normal(0, .002, days)), "Close": close,
                       "High": close * 1.006, "Low": close * 0.994,
                       "Volume": rng.integers(1e6, 5e6, days)}, index=dates)
    return df, news
