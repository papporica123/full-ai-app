# stockpredict

Predicts a stock's next-N-day return from its price history plus news propagated through a causal graph.

    python -m stockpredict --ticker XOM                      # offline synthetic demo
    python -m stockpredict --ticker NVDA --source yfinance --news csv --news-csv headlines.csv
    pytest tests

How it works: headlines -> node shocks (oil, rates, taiwan, ...) -> propagated up to 3 edges through
`graph.py` -> per-company features `impact_direct`, `impact_hop1..3` -> ridge model with technicals,
walk-forward trained (no lookahead). Output: expected return, P(up), 80% range, and which news paths drove it.

Extend: add tickers/edges in `graph.py` (or pass your own `CausalGraph`), swap `news.extract_shocks` for an LLM/FinBERT classifier.
Real validation needs a timestamped historical news CSV (`date,text[,ticker]`); the synthetic data only tests plumbing.
