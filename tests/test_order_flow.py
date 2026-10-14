import numpy as np
import pandas as pd

from order_flow.strategy import OrderFlowParams, compute_indicators, rolling_entropy, run


def _bars(closes, highs=None, lows=None, start="2024-01-02 09:30"):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="America/New_York")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": highs if highs is not None else c + 1, "low": lows if lows is not None else c - 1,
         "close": c, "volume": 100.0},
        index=idx,
    )


def test_rolling_entropy_matches_definition():
    ent = rolling_entropy(np.array([1, 1, 2, 2, 3, 4], dtype=float), 4)
    assert np.isnan(ent[2])
    assert ent[3] == 1.0  # {1,1,2,2}: two equally likely values = 1 bit
    assert np.isclose(ent[5], 1.5)  # {2,2,3,4}: p = .5,.25,.25


def test_entropy_ma_bug_doubles_average():
    df = _bars(np.arange(200) % 7 + 100.0)
    buggy = compute_indicators(df, OrderFlowParams(fix_entropy_ma=False))
    fixed = compute_indicators(df, OrderFlowParams(fix_entropy_ma=True))
    pd.testing.assert_series_equal(buggy["entropy_ma"].dropna(), 2 * fixed["entropy_ma"].dropna(), check_names=False)


def test_clv_proxy_is_close_location():
    # close in top quarter of range -> bid imbalance; bottom quarter -> ask imbalance
    df = _bars([100.0, 100.0, 100.0], highs=[101, 101, 101], lows=[97, 97, 97])
    df["close"] = [100.9, 99.0, 97.1]
    ind = compute_indicators(df, OrderFlowParams())
    assert list(ind["bid_imb"]) == [True, False, False]
    assert list(ind["ask_imb"]) == [False, False, True]


def test_long_entry_and_take_profit():
    p = OrderFlowParams(sma_period=5, entropy_period=3, entropy_ma_period=6)  # buggy MA = 2x mean
    # rising, every close at the bar high -> bid imbalance above SMA; then +1% move exits
    closes = list(np.linspace(100, 101, 20)) + [103.0]
    df = _bars(closes, highs=np.array(closes), lows=np.array(closes) - 1)
    trades = run(df, p)
    assert trades and trades[0].direction == 1 and trades[0].branch == "follow"
    assert trades[0].reason == "tp" and trades[0].exit_price == 103.0
