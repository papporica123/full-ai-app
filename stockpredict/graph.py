"""Causal impact graph.

Nodes are world variables (oil, rates, taiwan, ...). A shock on a node means "this
variable moved up / got more intense". Edges carry a signed weight: how much of a
unit shock on `src` is transmitted to `dst`. Companies attach to nodes through
signed exposures (betas).

Impact levels for a company:
  hop 0 - news directly hits a node the company is exposed to
  hop 1 - one edge away (e.g. Middle East conflict -> oil -> XOM)
  hop 2 - two edges away (conflict -> oil -> inflation -> rates -> NVDA is hop 3)
  hop 3 - three edges away
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_HOPS = 3

# (src, dst, weight): effect of a +1 shock on src onto dst
DEFAULT_EDGES = [
    ("middle_east", "oil", 0.8), ("russia", "oil", 0.4), ("russia", "gas", 0.7),
    ("middle_east", "defense_spending", 0.5), ("russia", "defense_spending", 0.5),
    ("gas", "inflation", 0.3), ("oil", "inflation", 0.4), ("oil", "shipping", 0.4),
    ("shipping", "inflation", 0.2),
    ("taiwan", "semiconductor_supply", 0.9),  # +shock = tighter / disrupted supply
    ("trade_war", "china_demand", -0.4), ("trade_war", "inflation", 0.3),
    ("trade_war", "semiconductor_supply", 0.3),
    ("china_demand", "oil", 0.4), ("china_demand", "consumer", 0.2),
    ("inflation", "rates", 0.6), ("inflation", "consumer", -0.3),
    ("rates", "usd", 0.4), ("rates", "banks_stress", 0.2), ("rates", "consumer", -0.5),
    ("usd", "china_demand", -0.2),
    ("banks_stress", "consumer", -0.4), ("banks_stress", "rates", -0.3),
]

# ticker -> {node: signed beta}. Override/extend via Predictor(exposures=...).
DEFAULT_EXPOSURES = {
    "XOM": {"oil": 0.8, "gas": 0.4, "consumer": 0.1},
    "DAL": {"oil": -0.6, "consumer": 0.5},
    "NVDA": {"semiconductor_supply": -0.4, "rates": -0.5, "china_demand": 0.3, "consumer": 0.2},
    "JPM": {"rates": 0.3, "banks_stress": -0.6, "consumer": 0.3},
    "LMT": {"defense_spending": 0.9, "middle_east": 0.2},
    "AAPL": {"china_demand": 0.4, "semiconductor_supply": -0.3, "trade_war": -0.4,
             "consumer": 0.4, "usd": -0.2},
    "WMT": {"consumer": 0.3, "inflation": -0.2, "shipping": -0.3},
}


@dataclass
class CausalGraph:
    edges: list = field(default_factory=lambda: list(DEFAULT_EDGES))
    exposures: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_EXPOSURES.items()})
    decay: float = 0.7  # attenuation per hop (uncertainty grows with distance)

    def __post_init__(self):
        names = {n for e in self.edges for n in e[:2]}
        for exp in self.exposures.values():
            names |= set(exp)
        self.nodes = sorted(names)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        self.A = np.zeros((n, n))  # A[dst, src]
        for s, d, w in self.edges:
            self.A[self.idx[d], self.idx[s]] += w * self.decay
        self._hop = [np.linalg.matrix_power(self.A, h) for h in range(MAX_HOPS + 1)]

    def exposure_vector(self, ticker: str) -> np.ndarray:
        v = np.zeros(len(self.nodes))
        for node, beta in self.exposures.get(ticker, {}).items():
            if node in self.idx:
                v[self.idx[node]] = beta
        return v

    def shock_vector(self, shocks: dict) -> np.ndarray:
        v = np.zeros(len(self.nodes))
        for node, val in shocks.items():
            if node in self.idx:
                v[self.idx[node]] += val
        return v

    def impact_by_hop(self, ticker: str, shock_state: np.ndarray) -> np.ndarray:
        """shock_state: (T, n_nodes) -> (T, MAX_HOPS+1) company impact per hop level."""
        e = self.exposure_vector(ticker)
        return np.stack([shock_state @ (e @ self._hop[h]) for h in range(MAX_HOPS + 1)], axis=1)

    def explain(self, ticker: str, shock_vec: np.ndarray, top: int = 6):
        """Rank (hop, source node -> ... ) contributions to the company impact."""
        e = self.exposure_vector(ticker)
        rows = []
        for h in range(MAX_HOPS + 1):
            contrib = shock_vec * (e @ self._hop[h])
            for i in np.argsort(-np.abs(contrib))[:top]:
                if abs(contrib[i]) > 1e-9:
                    rows.append((h, self.nodes[i], float(contrib[i])))
        return sorted(rows, key=lambda r: -abs(r[2]))[:top]


def default_graph() -> CausalGraph:
    return CausalGraph()
