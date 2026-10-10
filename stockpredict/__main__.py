"""python -m stockpredict --ticker XOM [--source synthetic|yfinance|csv] [--news none|rss|csv]"""
import argparse

from . import prices
from .backtest import evaluate
from .graph import default_graph
from .model import Predictor
from .news import WORLD_QUERIES, fetch_rss, load_csv as load_news_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="XOM")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "yfinance", "csv"])
    ap.add_argument("--prices-csv")
    ap.add_argument("--news", default="auto", choices=["auto", "none", "rss", "csv"])
    ap.add_argument("--news-csv")
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()

    g = default_graph()
    news = []
    if a.source == "synthetic":
        df, news = prices.synthetic(a.ticker, g)
        print("[synthetic data: demonstrates the pipeline, not real predictive power]")
    elif a.source == "yfinance":
        df = prices.load_yfinance(a.ticker)
    else:
        df = prices.load_csv(a.prices_csv)
    if a.news == "csv":
        news = load_news_csv(a.news_csv)
    elif a.news == "rss" or (a.news == "auto" and a.source != "synthetic"):
        for q in WORLD_QUERIES:
            news += fetch_rss(q)
        news += fetch_rss(f"{a.ticker} stock", a.ticker)
        print(f"fetched {len(news)} live headlines (RSS has little history; for training use --news csv)")

    p = Predictor(a.ticker, g, horizon=a.horizon)
    preds, realized = p.walk_forward(df, news)
    print("walk-forward out-of-sample:", evaluate(preds, realized, a.horizon))
    r = p.predict(df, news)
    print(f"\n{r.ticker} next {r.horizon}d: expected {r.expected_return:+.2%}  P(up)={r.prob_up:.0%}  "
          f"80% range [{r.interval80[0]:+.1%}, {r.interval80[1]:+.1%}]")
    print("top drivers:", ", ".join(f"{k} {v:+.4f}" for k, v in r.drivers))
    print("news transmission (hop, source node, contribution):")
    for hop, node, c in r.news_paths:
        print(f"  hop{hop}  {node:<22}{c:+.3f}")
    print("\nNot financial advice.")


main()
