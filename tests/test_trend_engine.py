import numpy as np

from trend import engine


def _run(o, h, l, c, long_sig, regime=None, sl_k=1.0, tp_k=2.0, risk=100.0, small=0.5, flatten=None,
         fee=2.5, slip=1.0, dll=1e9, cap=1e9, flip=False, trend=None, trail=0.0):
    n = len(c)
    atr = np.full(n, 4.0)
    regime = np.full(n, 2.0) if regime is None else np.asarray(regime, float)
    flatten = np.zeros(n, bool) if flatten is None else np.asarray(flatten, bool)
    trend = np.ones(n) if trend is None else np.asarray(trend, float)
    return engine.run(np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), np.asarray(c, float), atr,
                      np.asarray(long_sig, bool), np.zeros(n, bool), trend, regime, np.ones(n, bool), flatten,
                      np.zeros(n, np.int64), sl_k, tp_k, risk, small, 5.0, 0.25, fee, slip, 40, dll, cap, flip, trail)


def test_long_target_hit_next_bar_fill_and_fees():
    # signal on bar 0 close; fill at bar 1 open 100 + 1 tick slip = 100.25; target = +2*4 = 108.25
    o = [100, 100, 104, 107]
    h = [100, 101, 105, 109]
    l = [99, 99, 103, 106]
    c = [100, 100, 104, 108]
    tr = _run(o, h, l, c, [1, 0, 0, 0])
    assert len(tr) == 1
    t = tr[0]
    assert t[engine.T_ENTRY_I] == 1 and t[engine.T_EXIT_I] == 3
    assert t[engine.T_ENTRY_PX] == 100.25
    assert t[engine.T_EXIT_PX] == 108.25
    # qty = 100 / (1*4*5 + 2.5 + 2*0.25*5) = 100/25 = 4
    assert t[engine.T_QTY] == 4
    assert np.isclose(t[engine.T_PNL], 8.0 * 5 * 4 - 2.5 * 4)
    assert engine.REASONS[int(t[engine.T_REASON])] == "target"


def test_stop_assumed_first_when_bar_spans_both():
    o = [100, 100, 100]
    h = [100, 100.5, 120]
    l = [99, 99.5, 80]
    c = [100, 100, 100]
    t = _run(o, h, l, c, [1, 0, 0])[0]
    assert engine.REASONS[int(t[engine.T_REASON])] == "stop"
    assert t[engine.T_EXIT_PX] == 100.25 - 4.0 - 0.25


def test_regime_zero_blocks_and_weak_regime_halves_size():
    o = h = l = c = [100.0] * 6
    assert len(_run(o, h, l, c, [1, 0, 0, 0, 0, 0], regime=[0] * 6)) == 0
    flat = [0, 0, 1, 0, 0, 0]
    tr = _run(o, [101] * 6, [99] * 6, c, [1, 0, 0, 0, 0, 0], regime=[1] * 6, flatten=flat)
    assert tr[0][engine.T_QTY] == 2  # int(100*0.5/25)


def test_flatten_at_session_cutoff():
    o = h = l = c = [100.0] * 5
    tr = _run(o, [100.5] * 5, [99.5] * 5, c, [1, 0, 0, 0, 0], flatten=[0, 0, 0, 1, 0])
    assert len(tr) == 1 and tr[0][engine.T_EXIT_I] == 3
    assert engine.REASONS[int(tr[0][engine.T_REASON])] == "flatten"


def test_daily_loss_limit_locks_day():
    n = 8
    o = [100.0] * n
    h = [100.0] * n
    l = [90.0] * n  # every entry stops out immediately
    c = [100.0] * n
    tr = _run(o, h, l, c, [1] * n, dll=50.0)
    assert len(tr) == 1


def test_trailing_stop_ratchets():
    # fill 100.25, atr 4, trail 1.0 -> after bar 2 high 110 stop = 106; bar 3 low 105 stops at 106 - slip
    o = [100, 100, 108, 107]
    h = [100, 101, 110, 108]
    l = [99, 99, 107, 105]
    c = [100, 100, 109, 106]
    t = _run(o, h, l, c, [1, 0, 0, 0], tp_k=10.0, trail=1.0)[0]
    assert engine.REASONS[int(t[engine.T_REASON])] == "stop"
    assert t[engine.T_EXIT_I] == 3 and t[engine.T_EXIT_PX] == 106.0 - 0.25
