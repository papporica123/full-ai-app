"""News -> node shocks. Rule/lexicon based so it is transparent and testable.

Swap `extract_shocks` for an LLM or FinBERT classifier if you want richer parsing;
the rest of the pipeline only needs {node: signed_magnitude}.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

ESCALATE = r"(war|attack|missile|strike|strikes|sanction|sanctions|tension|tensions|escalat\w*|invasion|clash\w*|blockade|drone)"
DEESCALATE = r"(ceasefire|truce|peace|de-escalat\w*|talks|deal reached)"
UP = r"(surge\w*|jump\w*|soar\w*|rise|rises|rising|climb\w*|spike\w*|rall(y|ies)|hot(ter)?|higher|hike\w*)"
DOWN = r"(fall\w*|drop\w*|plunge\w*|slump\w*|tumble\w*|slide\w*|decline\w*|cool\w*|lower|cut\w*|ease\w*)"

REGIONS = {
    "middle_east": r"(israel|iran|gaza|saudi|houthi|red sea|hormuz|lebanon|yemen|middle east)",
    "russia": r"(russia\w*|ukrain\w*|kremlin|moscow)",
    "taiwan": r"(taiwan\w*|tsmc|taipei)",
}
# (node, regex, sign)
VARIABLE_RULES = [
    ("oil", rf"(oil|crude|brent|opec).*{UP}|{UP}.*(oil|crude|brent)", +1),
    ("oil", rf"(oil|crude|brent|opec).*{DOWN}|{DOWN}.*(oil|crude|brent)", -1),
    ("gas", rf"(natural gas|lng).*{UP}", +1),
    ("gas", rf"(natural gas|lng).*{DOWN}", -1),
    ("inflation", rf"(inflation|cpi|prices).*{UP}", +1),
    ("inflation", rf"(inflation|cpi).*{DOWN}", -1),
    ("rates", r"(rate hike|raises rates|hawkish|rates? (higher|rise))", +1),
    ("rates", r"(rate cut|cuts rates|dovish|pivot)", -1),
    ("usd", rf"(dollar|usd).*{UP}", +1),
    ("usd", rf"(dollar|usd).*{DOWN}", -1),
    ("china_demand", r"china.*(slowdown|weak\w*|contract\w*|deflation|property crisis)", -1),
    ("china_demand", r"china.*(stimulus|recovery|rebound|reopen\w*|strong\w*)", +1),
    ("trade_war", r"(tariff\w*|trade war|export controls?|embargo)", +1),
    ("trade_war", r"(tariffs? (removed|lifted|eased)|trade deal)", -1),
    ("semiconductor_supply", r"(chip|semiconductor)s?.*(shortage|disruption|constraint\w*)", +1),
    ("semiconductor_supply", r"(chip|semiconductor)s?.*(glut|oversupply|recover\w*)", -1),
    ("shipping", r"(port|shipping|freight|supply chain).*(strike|delay\w*|disrupt\w*|congest\w*|surge\w*)", +1),
    ("banks_stress", r"(bank (run|failure|collapse)|credit crunch|bank.*(default|insolven\w*))", +1),
    ("consumer", r"(consumer (spending|confidence)|retail sales).*(strong\w*|rise|rises|jump\w*|beat\w*)", +1),
    ("consumer", r"(consumer (spending|confidence)|retail sales).*(weak\w*|fall\w*|drop\w*|miss\w*)", -1),
    ("defense_spending", r"(defen[cs]e (budget|spending)|military aid|arms).*(increase\w*|boost\w*|surge\w*|rise)", +1),
]
INTENSIFIERS = {r"(surge|soar|plunge|collapse|crisis|record|spike|crash)": 1.5, r"(slight|modest|mild|edge)": 0.5}
POS = r"(beat\w*|record|upgrade\w*|raises guidance|strong|growth|profit|wins?|approval|buyback|outperform\w*)"
NEG = r"(miss\w*|downgrade\w*|lawsuit|probe|recall|cuts guidance|weak|loss|layoff\w*|fraud|bankrupt\w*|investigation)"


@dataclass
class Headline:
    ts: pd.Timestamp
    text: str
    ticker: str | None = None  # set when the item is about a specific company
    shocks: dict | None = None  # precomputed node shocks (e.g. from GDELT); skips text extraction


def extract_shocks(text: str) -> dict:
    t = text.lower()
    mag = 1.0
    for pat, m in INTENSIFIERS.items():
        if re.search(pat, t):
            mag = m
    shocks: dict = {}
    for node, pat in REGIONS.items():
        if re.search(pat, t):
            if re.search(DEESCALATE, t):
                shocks[node] = shocks.get(node, 0) - mag
            elif re.search(ESCALATE, t):
                shocks[node] = shocks.get(node, 0) + mag
    for node, pat, sign in VARIABLE_RULES:
        if re.search(pat, t):
            shocks[node] = shocks.get(node, 0) + sign * mag
    return shocks


def sentiment(text: str) -> float:
    t = text.lower()
    return float(bool(re.search(POS, t))) - float(bool(re.search(NEG, t)))


def load_csv(path: str) -> list[Headline]:
    """Historical news CSV with columns: date,text[,ticker]. Needed for honest backtests."""
    df = pd.read_csv(path, parse_dates=["date"])
    return [Headline(r.date, r.text, getattr(r, "ticker", None) if "ticker" in df else None)
            for r in df.itertuples()]


def fetch_rss(query: str, ticker: str | None = None, timeout: int = 15) -> list[Headline]:
    """Live headlines from Google News RSS (recent items only; no deep history)."""
    import requests

    url = "https://news.google.com/rss/search"
    r = requests.get(url, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}, timeout=timeout)
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).iter("item"):
        title, pub = item.findtext("title", ""), item.findtext("pubDate")
        try:
            ts = pd.Timestamp(datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")).normalize()
        except (TypeError, ValueError):
            continue
        out.append(Headline(ts, title, ticker))
    return out


WORLD_QUERIES = ["oil prices", "Federal Reserve interest rates", "inflation CPI", "Taiwan semiconductor",
                 "Russia Ukraine", "Israel Iran Middle East", "China economy", "tariffs trade war",
                 "bank stress", "shipping supply chain"]
