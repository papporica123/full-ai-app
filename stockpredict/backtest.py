"""Walk-forward evaluation with honest baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(preds: pd.Series, realized: pd.Series, horizon: int, cost_bps: float = 5.0) -> dict:
    d = pd.DataFrame({"p": preds, "y": realized}).dropna()
    if len(d) < 30:
        return {"n": len(d), "note": "too few out-of-sample points"}
    ic = d.p.rank().corr(d.y.rank())
    hit = float((np.sign(d.p) == np.sign(d.y)).mean())
    base = float((d.y > 0).mean())  # accuracy of "always up"
    pos = np.sign(d.p).values
    # non-overlapping H-day trades to avoid double counting
    sel = slice(None, None, horizon)
    trade = pos[sel] * d.y.values[sel] - cost_bps / 1e4 * np.abs(np.diff(np.r_[0, pos[sel]]))
    sharpe = trade.mean() / (trade.std() + 1e-12) * np.sqrt(252 / horizon)
    return {"n": len(d), "rank_IC": round(float(ic), 3), "hit_rate": round(hit, 3),
            "always_up_baseline": round(base, 3), "strategy_sharpe_net": round(float(sharpe), 2),
            "buy_hold_sharpe": round(float(d.y.values[sel].mean() / (d.y.values[sel].std() + 1e-12)
                                           * np.sqrt(252 / horizon)), 2)}
