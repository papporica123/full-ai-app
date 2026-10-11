import numpy as np
import pandas as pd
from stockpredict import default_graph
from stockpredict.gdelt import daily_shocks, events_sql, gkg_sql, load_gdelt_headlines, to_headlines
from stockpredict.features import shock_state


def fake_runner(spike_day):
    days = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(0)

    def run(sql, params):
        if "events_partitioned" in sql:
            m = rng.integers(900, 1100, len(days)).astype(float)
            m[days.get_loc(spike_day)] = 30000  # big Middle East escalation
            return pd.DataFrame({"day": days, "middle_east_mentions": m, "middle_east_goldstein": -3.0,
                                 "russia_mentions": 1000.0, "russia_goldstein": -3.0,
                                 "taiwan_mentions": 50.0, "taiwan_goldstein": -1.0})
        return pd.DataFrame({"day": days, "total_articles": 100000, "trade_war_articles": 500,
                             "trade_war_tone": -2.0, "inflation_articles": 800, "inflation_tone": -1.0,
                             "banks_stress_articles": 100, "banks_stress_tone": -3.0})
    return run


def test_sql_is_parameterised():
    for q in (events_sql(), gkg_sql()):
        assert "@start" in q and "@end" in q and "_PARTITIONTIME" in q


def test_spike_becomes_positive_shock_with_lag_and_flows_through_graph():
    spike = pd.Timestamp("2025-09-01")
    heads = load_gdelt_headlines("2025-06-01", "2025-10-15", runner=fake_runner(spike))
    hit = [h for h in heads if h.shocks.get("middle_east", 0) > 0.9]  # 0.7*z(3)*scale 0.5 = 1.05
    assert [h.ts for h in hit] == [spike + pd.Timedelta(days=1)]  # lagged: no lookahead
    g = default_graph()
    dates = pd.bdate_range("2025-08-25", "2025-09-12")
    imp = g.impact_by_hop("XOM", shock_state(heads, dates, g))
    assert imp[:, 1].max() > 0  # conflict -> oil -> XOM, hop 1


def test_quiet_baseline_produces_few_shocks():
    sh = daily_shocks(fake_runner(pd.Timestamp("2025-09-01"))("events_partitioned", {}), None)
    assert (sh["russia"].abs() < 0.25).mean() > 0.9
