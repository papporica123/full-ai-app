import numpy as np
from stockpredict import default_graph, Predictor
from stockpredict.news import extract_shocks
from stockpredict import prices


def test_extraction():
    assert extract_shocks("Missile attack escalates tension in Israel and Iran")["middle_east"] > 0
    assert extract_shocks("Ceasefire talks calm Gaza conflict")["middle_east"] < 0
    assert extract_shocks("Fed hints at rate cut")["rates"] < 0


def test_multi_hop_signs():
    g = default_graph()
    s = g.shock_vector({"middle_east": 1.0})
    xom, dal = g.impact_by_hop("XOM", s[None]), g.impact_by_hop("DAL", s[None])
    assert xom[0, 1] > 0       # conflict -> oil -> XOM (hop 1)
    assert dal[0, 1] < 0       # conflict -> oil -> airline fuel costs
    assert abs(xom[0, 3]) > 0  # third-order effect exists
    assert g.impact_by_hop("XOM", s[None]).shape == (1, 4)


def test_no_lookahead_and_signal():
    g = default_graph()
    df, news = prices.synthetic("XOM", g, days=900, seed=1)
    p = Predictor("XOM", g)
    preds, y = p.walk_forward(df, news)
    assert preds.iloc[:250].isna().all()
    d = preds.to_frame("p").join(y.rename("y")).dropna()
    assert d.p.corr(d.y) > 0.05  # the injected multi-hop signal is recovered
    r = p.predict(df, news)
    assert 0 <= r.prob_up <= 1
