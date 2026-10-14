import numpy as np
import pytest

pytest.importorskip("numba")

from ppo_dd.sim import R_MAXQTY, R_NET, R_TRADES, R_WINS, simulate


def _run(o, h, l, c, *, max_dd=0, mults=0.0, dd=1.0):
    n = len(o)
    o, h, l, c = (np.asarray(x, dtype=float) for x in (o, h, l, c))
    t = np.arange(n, dtype=np.int64) * 60_000
    day = np.zeros(n, dtype=np.int64)
    ema = np.full(n, 1e9)  # EMA far above price -> long signal on every flat bar
    return simulate(o, h, l, c, t, day, ema, ema, 1, 1.0, 1.0, 1, 0.01, max_dd, 0.5, True, False,
                    np.full(10, dd / 100), np.full(10, 0.01), np.full(10, mults),
                    np.zeros(11), np.full(11, 0.0001), False, 1.0, 0.0, 0.0, 0, n, np.zeros((0, 6)))


def test_entry_next_open_then_tp_limit_fill():
    # Signal on bar 0 -> fill bar 1 open 100 -> TP 101 live from bar 2.
    r = _run(o=[100, 100, 100], h=[100, 100.5, 101.5], l=[100, 99.5, 99.5], c=[100, 100, 100])
    assert r[R_TRADES] == 1 and r[R_WINS] == 1
    assert r[R_NET] == pytest.approx(1.0)


def test_dd_add_fills_next_open_and_doubles():
    # Fill at 100, DD trigger at 99 hit on bar 2 -> add 1 lot at bar 3 open (98).
    r = _run(o=[100, 100, 99.5, 98, 98], h=[100, 100, 99.5, 98, 98], l=[100, 100, 98.5, 98, 98],
             c=[100, 100, 99, 98, 98], max_dd=1, mults=1.0)
    assert r[R_MAXQTY] == 2
