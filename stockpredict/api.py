"""REST API.  Run: uvicorn stockpredict.api:app --reload   (docs at /docs)

Data source: STOCK_SOURCE=yfinance (default; falls back to synthetic, flagged in the
response) or STOCK_SOURCE=synthetic. Optional auth: set API_KEY and send X-API-Key.
"""
from __future__ import annotations

import os
import re
import time

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from . import prices
from .features import technicals
from .graph import default_graph
from .model import Predictor
from .news import WORLD_QUERIES, fetch_rss

app = FastAPI(title="stockpredict API", version="0.1.0")
graph = default_graph()
_cache: dict = {}
TTL = int(os.getenv("CACHE_TTL", "300"))
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,12}$")


def auth(x_api_key: str | None = Header(default=None)):
    key = os.getenv("API_KEY")
    if key and x_api_key != key:
        raise HTTPException(401, "invalid or missing X-API-Key")


def _cached(key, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def _ticker(t: str) -> str:
    if not TICKER_RE.match(t):
        raise HTTPException(422, "invalid ticker")
    return t.upper()


def _load(ticker: str):
    """Returns (df, news, source). Never silently passes synthetic data off as real."""
    def go():
        if os.getenv("STOCK_SOURCE", "yfinance") == "yfinance":
            try:
                df = prices.load_yfinance(ticker)
                if len(df):
                    return df, [], "yfinance"
            except Exception:
                pass
        df, news = prices.synthetic(ticker, graph)
        return df, news, "synthetic"
    return _cached(("px", ticker), go)


def _live_news(ticker: str):
    def go():
        out = []
        for q in WORLD_QUERIES + [f"{ticker} stock"]:
            try:
                out += fetch_rss(q, ticker if q.endswith("stock") else None)
            except Exception:
                continue
        return out
    return _cached(("news", ticker), go)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stocks/{ticker}/prices", dependencies=[Depends(auth)])
def get_prices(ticker: str, start: str | None = None, end: str | None = None,
               limit: int = Query(250, ge=1, le=5000), indicators: bool = False):
    t = _ticker(ticker)
    df, _, source = _load(t)
    if indicators:
        df = df.join(technicals(df))
    try:
        df = df.loc[start:end]
    except Exception:
        raise HTTPException(422, "bad start/end date (use YYYY-MM-DD)")
    df = df.tail(limit)
    if df.empty:
        raise HTTPException(404, "no data for that range")
    rows = df.reset_index().rename(columns={df.index.name or "index": "date"})
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    return {"ticker": t, "source": source, "count": len(rows),
            "data": rows.round(6).where(rows.notna(), None).to_dict("records")}


@app.get("/stocks/{ticker}/predict", dependencies=[Depends(auth)])
def predict(ticker: str, horizon: int = Query(5, ge=1, le=30)):
    t = _ticker(ticker)
    df, news, source = _load(t)
    if source == "yfinance":
        news = _live_news(t)
    r = Predictor(t, graph, horizon=horizon).predict(df, news)
    return {"ticker": t, "source": source, "horizon_days": horizon,
            "expected_return": r.expected_return, "prob_up": r.prob_up,
            "range_80": list(r.interval80),
            "drivers": [{"feature": k, "contribution": v} for k, v in r.drivers],
            "news_paths": [{"hop": h, "node": n, "contribution": c} for h, n, c in r.news_paths],
            "warning": "synthetic data - demo only" if source == "synthetic" else
                       "live RSS news has little history; not financial advice"}
