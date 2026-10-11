import os
os.environ["STOCK_SOURCE"] = "synthetic"
from fastapi.testclient import TestClient
from stockpredict.api import app

c = TestClient(app)


def test_prices_and_predict():
    r = c.get("/stocks/xom/prices?limit=5&indicators=true").json()
    assert r["ticker"] == "XOM" and r["source"] == "synthetic" and r["count"] == 5
    p = c.get("/stocks/XOM/predict?horizon=3").json()
    assert 0 <= p["prob_up"] <= 1 and p["horizon_days"] == 3


def test_validation_and_auth(monkeypatch):
    assert c.get("/stocks/bad$ticker/prices").status_code in (404, 422)
    assert c.get("/stocks/XOM/prices?start=garbage").status_code == 422
    monkeypatch.setenv("API_KEY", "k")
    assert c.get("/stocks/XOM/prices").status_code == 401
    assert c.get("/stocks/XOM/prices", headers={"X-API-Key": "k"}).status_code == 200
