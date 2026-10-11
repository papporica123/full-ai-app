"""GDELT (via Google BigQuery) -> historical node shocks.

Two public tables are used:
  gdelt-bq.gdeltv2.events_partitioned - CAMEO-coded events; used for conflict nodes
      (middle_east, russia, taiwan): daily mentions of violent events (CAMEO roots 18/19/20)
      in the region, plus Goldstein score. Small and cheap to scan.
  gdelt-bq.gdeltv2.gkg_partitioned   - themes + tone per article; used for attention nodes
      (trade_war, inflation, banks_stress). LARGE (hundreds of GB per year): always --dry-run first.

Signs: GDELT says how much the world is talking about / fighting over something, not which
way a price moved. So only nodes with a natural sign are produced (escalation, trade-war
intensity, inflation worry, bank stress, all "more = +"). Oil/rates/usd direction should
come from market data, not GDELT.

Daily value -> shock = scale * z-score vs the trailing `baseline` days (excluding today),
and the headline is dated day+lag_days (default 1) because a GDELT day is only complete
after it ends: using it on the same close would be lookahead.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd

from .news import Headline

EVENTS_TABLE = "gdelt-bq.gdeltv2.events_partitioned"
GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

# GDELT uses FIPS country codes
CONFLICT_REGIONS = {
    "middle_east": ["IS", "IR", "IZ", "SA", "YM", "LE", "SY", "GZ", "WE"],
    "russia": ["RS", "UP"],
    "taiwan": ["TW"],
}
# Theme names follow GDELT's GKG taxonomy; verify against your data with `--dry-run`/sample
# queries and tune. Edit freely - this is configuration, not logic.
ATTENTION_THEMES = {
    "trade_war": r"ECON_TRADE_DISPUTE|TRADE_DISPUTE|TARIFF|EMBARGO|SANCTION",
    "inflation": r"ECON_INFLATION|INFLATION",
    "banks_stress": r"ECON_BANKRUPTCY|BANKRUPTCY|BANK_RUN|ECON_DEBT|CREDIT_CRUNCH",
}


def events_sql() -> str:
    cases = ",\n  ".join(
        f"SUM(IF(ActionGeo_CountryCode IN ({','.join(repr(c) for c in codes)}), NumMentions, 0)) AS {n}_mentions,\n  "
        f"AVG(IF(ActionGeo_CountryCode IN ({','.join(repr(c) for c in codes)}), GoldsteinScale, NULL)) AS {n}_goldstein"
        for n, codes in CONFLICT_REGIONS.items())
    return f"""SELECT PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS day,
  {cases}
FROM `{EVENTS_TABLE}`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start) AND TIMESTAMP(@end)
  AND SQLDATE BETWEEN @start_int AND @end_int
  AND EventRootCode IN ('18','19','20')
GROUP BY day ORDER BY day"""


def gkg_sql() -> str:
    cols = ",\n  ".join(
        f"COUNTIF(REGEXP_CONTAINS(V2Themes, r'{pat}')) AS {n}_articles,\n  "
        f"AVG(IF(REGEXP_CONTAINS(V2Themes, r'{pat}'), CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64), NULL)) AS {n}_tone"
        for n, pat in ATTENTION_THEMES.items())
    return f"""SELECT PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS day,
  COUNT(*) AS total_articles,
  {cols}
FROM `{GKG_TABLE}`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start) AND TIMESTAMP(@end)
GROUP BY day ORDER BY day"""


class BigQueryRunner:
    """Runs parameterised SQL with a hard byte cap and an on-disk cache."""

    def __init__(self, project: str | None = None, max_gb: float = 200.0, cache_dir: str = ".gdelt_cache"):
        self.project = project or os.getenv("GDELT_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.max_bytes, self.cache_dir = int(max_gb * 1e9), cache_dir
        if not self.project:
            raise RuntimeError("Set a Google Cloud project (GDELT_PROJECT or project=...) and credentials "
                               "(gcloud auth application-default login).")

    def _job_config(self, params, dry_run):
        from google.cloud import bigquery as bq
        qp = [bq.ScalarQueryParameter(k, "INT64" if isinstance(v, int) else "STRING", v) for k, v in params.items()]
        return bq.QueryJobConfig(query_parameters=qp, dry_run=dry_run, use_query_cache=True,
                                 maximum_bytes_billed=self.max_bytes)

    def dry_run(self, sql, params) -> float:
        from google.cloud import bigquery as bq
        job = bq.Client(project=self.project).query(sql, job_config=self._job_config(params, True))
        return job.total_bytes_processed / 1e9

    def __call__(self, sql, params) -> pd.DataFrame:
        from google.cloud import bigquery as bq
        key = hashlib.sha256((sql + repr(sorted(params.items()))).encode()).hexdigest()[:16]
        path = os.path.join(self.cache_dir, f"{key}.csv")
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["day"])
        df = bq.Client(project=self.project).query(sql, job_config=self._job_config(params, False)).to_dataframe()
        df["day"] = pd.to_datetime(df["day"])
        os.makedirs(self.cache_dir, exist_ok=True)
        df.to_csv(path, index=False)
        return df


def _zscore(s: pd.Series, baseline: int) -> pd.Series:
    past = s.shift(1)  # trailing stats exclude today
    mu = past.rolling(baseline, min_periods=max(20, baseline // 3)).mean()
    sd = past.rolling(baseline, min_periods=max(20, baseline // 3)).std()
    return ((s - mu) / (sd + 1e-9)).clip(-3, 3)


def daily_shocks(events: pd.DataFrame | None, gkg: pd.DataFrame | None, baseline: int = 90) -> pd.DataFrame:
    """DataFrame indexed by day, one column per node, values = z-scored pressure (+ = more intense)."""
    out = {}
    if events is not None and len(events):
        e = events.set_index("day").sort_index()
        for node in CONFLICT_REGIONS:
            if f"{node}_mentions" in e:
                vol = np.log1p(e[f"{node}_mentions"].fillna(0))
                gold = -e[f"{node}_goldstein"].fillna(0)  # more negative Goldstein = more hostile
                out[node] = 0.7 * _zscore(vol, baseline) + 0.3 * _zscore(gold, baseline)
    if gkg is not None and len(gkg):
        g = gkg.set_index("day").sort_index()
        for node in ATTENTION_THEMES:
            if f"{node}_articles" in g:
                share = g[f"{node}_articles"] / g["total_articles"].clip(lower=1)  # normalise by news volume
                z = _zscore(share, baseline)
                if node == "banks_stress":  # stress needs bad tone too, not just attention
                    z = 0.6 * z + 0.4 * _zscore(-g[f"{node}_tone"].fillna(0), baseline)
                out[node] = z
    return pd.DataFrame(out).dropna(how="all").fillna(0.0)


def to_headlines(shocks: pd.DataFrame, scale: float = 0.5, lag_days: int = 1,
                 min_abs: float = 0.75) -> list[Headline]:
    heads = []
    for day, row in shocks.iterrows():
        s = {n: float(v) * scale for n, v in row.items() if abs(v) >= min_abs}
        if s:
            heads.append(Headline(pd.Timestamp(day) + pd.Timedelta(days=lag_days),
                                  f"GDELT daily pressure {pd.Timestamp(day).date()}", None, s))
    return heads


def load_gdelt_headlines(start, end, runner=None, project=None, use_gkg: bool = True, baseline: int = 90,
                         scale: float = 0.5, lag_days: int = 1) -> list[Headline]:
    """Fetch GDELT for [start, end] (plus a `baseline`-day warm-up for the z-scores)."""
    runner = runner or BigQueryRunner(project)
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    warm = s - pd.Timedelta(days=int(baseline * 1.5))
    params = {"start": warm.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d"),
              "start_int": int(warm.strftime("%Y%m%d")), "end_int": int(e.strftime("%Y%m%d"))}
    ev = runner(events_sql(), params)
    gk = runner(gkg_sql(), {k: params[k] for k in ("start", "end")}) if use_gkg else None
    sh = daily_shocks(ev, gk, baseline)
    return to_headlines(sh.loc[s - pd.Timedelta(days=lag_days):], scale, lag_days)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Check GDELT query cost or export daily shocks to CSV")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--project"); ap.add_argument("--no-gkg", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only print GB that would be scanned")
    ap.add_argument("--out", help="write daily shocks CSV")
    a = ap.parse_args()
    r = BigQueryRunner(a.project)
    s = pd.Timestamp(a.start) - pd.Timedelta(days=135); e = pd.Timestamp(a.end)
    p = {"start": s.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d"),
         "start_int": int(s.strftime("%Y%m%d")), "end_int": int(e.strftime("%Y%m%d"))}
    if a.dry_run:
        print(f"events: {r.dry_run(events_sql(), p):.2f} GB")
        if not a.no_gkg:
            print(f"gkg:    {r.dry_run(gkg_sql(), {k: p[k] for k in ('start', 'end')}):.2f} GB")
        return
    ev = r(events_sql(), p)
    gk = None if a.no_gkg else r(gkg_sql(), {k: p[k] for k in ("start", "end")})
    sh = daily_shocks(ev, gk)
    print(sh.describe().round(2))
    if a.out:
        sh.to_csv(a.out)


if __name__ == "__main__":
    main()
