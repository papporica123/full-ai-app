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

## API

    uvicorn stockpredict.api:app          # interactive docs at /docs
    GET /stocks/{ticker}/prices?start=2025-01-01&end=2025-06-30&limit=250&indicators=true
    GET /stocks/{ticker}/predict?horizon=5
    GET /health

Env: `STOCK_SOURCE=yfinance|synthetic`, `API_KEY` (require `X-API-Key` header), `CACHE_TTL` seconds.
Responses include `source`; if live data can't be fetched the API falls back to `synthetic` and says so.

## Historical news from GDELT (BigQuery)

    pip install google-cloud-bigquery db-dtypes && gcloud auth application-default login
    export GDELT_PROJECT=<your-gcp-project>
    python -m stockpredict.gdelt --start 2023-01-01 --end 2025-12-31 --dry-run   # GB scanned - check first!
    python -m stockpredict --ticker XOM --source yfinance --news gdelt

Conflict nodes (middle_east, russia, taiwan) come from the GDELT events table (cheap); trade_war, inflation and
banks_stress come from the GKG themes table (large - use `--no-gkg` or short ranges to limit cost; queries are capped at
200 GB billed and cached in `.gdelt_cache/`). Daily values are z-scored against the trailing 90 days and applied with a
1-day lag. GDELT has no price direction, so oil/rates/usd are not taken from it. Theme names in `gdelt.py` should be
checked against your data.
