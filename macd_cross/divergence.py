"""
MACD Strategy 4: price / MACD divergence.

- Swing low at bar j: low[j] is the lowest low of bars j-k .. j+k. It is only
  *known* at bar j+k, so that's when it can be used (no lookahead). Swing
  highs mirror this.
- Bullish (positive) divergence: a swing low makes a lower low than the
  previous swing low (min_gap..max_gap bars earlier) while the MACD line
  at the new swing is higher than at the previous one. Bearish mirrors it.

Entries (`confirm`):
- False ("raw"): enter at the next bar's open once the divergence is
  confirmed.
- True ("confirmed"): the divergence only arms a setup; enter on the first
  MACD/signal-line cross in the divergence direction within `confirm_window`
  bars of confirmation.

Exits: stop 1 tick beyond the divergence swing extreme (fills at the stop,
or the bar's open if it gaps through), the opposite MACD/signal cross (next
bar's open), or the RTH flatten time. New entries only while flat and only
from signal bars closing inside the RTH entry window.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from failed2s.instruments import Instrument

from .backtest import CostModel, SessionRules, Trade
from .strategy import MACDParams, crossover_signals, macd


@dataclass(frozen=True)
class DivergenceParams:
    pivot_k: int = 3
    min_gap: int = 5
    max_gap: int = 60
    confirm: bool = False
    confirm_window: int = 10


def find_divergences(bars: pd.DataFrame, params: MACDParams = MACDParams(),
                     div: DivergenceParams = DivergenceParams()):
    """
    Returns (signal, stop) arrays indexed by the bar on which the divergence
    becomes known: signal +1 bullish / -1 bearish / 0, stop = swing extreme.
    """
    k = div.pivot_k
    low, high = bars["low"].to_numpy(), bars["high"].to_numpy()
    m = macd(bars["close"], params)["macd"].to_numpy()
    n = len(bars)
    warmup = params.slow + params.signal
    win = 2 * k + 1
    roll_min = bars["low"].rolling(win, center=True).min().to_numpy()
    roll_max = bars["high"].rolling(win, center=True).max().to_numpy()

    signal = np.zeros(n, dtype=int)
    stop = np.full(n, np.nan)
    last_low: Optional[int] = None
    last_high: Optional[int] = None
    for j in range(warmup, n - k):
        c = j + k  # bar on which swing j is confirmed
        if low[j] == roll_min[j]:
            if last_low is not None and div.min_gap <= j - last_low <= div.max_gap \
                    and low[j] < low[last_low] and m[j] > m[last_low]:
                signal[c], stop[c] = 1, low[j]
            last_low = j
        if high[j] == roll_max[j]:
            if last_high is not None and div.min_gap <= j - last_high <= div.max_gap \
                    and high[j] > high[last_high] and m[j] < m[last_high]:
                signal[c], stop[c] = -1, high[j]
            last_high = j
    return signal, stop


def _entry_triggers(bars, params, div, cross):
    sig, stop = find_divergences(bars, params, div)
    if not div.confirm:
        return sig, stop
    n = len(bars)
    trig = np.zeros(n, dtype=int)
    tstop = np.full(n, np.nan)
    armed, armed_stop, armed_until = 0, np.nan, -1
    for i in range(n):
        if sig[i] != 0:
            armed, armed_stop, armed_until = sig[i], stop[i], i + div.confirm_window
        if armed and i <= armed_until and cross[i] == armed:
            trig[i], tstop[i] = armed, armed_stop
            armed = 0
        elif i > armed_until:
            armed = 0
    return trig, tstop


def run_divergence_backtest(
    bars: pd.DataFrame,
    bar_len: pd.Timedelta,
    inst: Instrument,
    params: MACDParams = MACDParams(),
    div: DivergenceParams = DivergenceParams(),
    costs: CostModel = CostModel(),
    rules: SessionRules = SessionRules(mode="rth"),
    contracts: int = 1,
) -> List[Trade]:
    cross = crossover_signals(bars["close"], params).to_numpy()
    trig, tstop = _entry_triggers(bars, params, div, cross)
    o, h, l, c = (bars[x].to_numpy() for x in ("open", "high", "low", "close"))
    idx = bars.index
    close_times = idx + bar_len
    rt_cost = costs.round_turn(inst, contracts)
    tick = inst.tick_size

    trades: List[Trade] = []
    pos, entry_px, stop_px, entry_t = 0, 0.0, 0.0, None

    def close_pos(px, t, reason):
        nonlocal pos
        pts = (px - entry_px) * pos
        gross = pts * inst.point_value * contracts
        trades.append(Trade(entry_t, t, pos, entry_px, px, reason, pts, gross, rt_cost, gross - rt_cost))
        pos = 0

    for i in range(1, len(bars)):
        t = idx[i]
        # --- at the open ---
        if pos != 0 and t.date() != entry_t.date():
            close_pos(c[i - 1], idx[i - 1] + bar_len, "session_flatten")
        if pos != 0 and t.time() >= rules.flatten_at:
            close_pos(o[i], t, "session_flatten")
        if pos != 0 and cross[i - 1] == -pos:
            close_pos(o[i], t, "signal_exit")
        if pos == 0 and trig[i - 1] != 0:
            ct = close_times[i - 1].time()
            if rules.entry_start < ct <= rules.entry_end and t.time() < rules.flatten_at \
                    and t.date() == idx[i - 1].date():
                d = trig[i - 1]
                sp = tstop[i - 1] - d * tick
                if (o[i] - sp) * d > 0:  # stop still on the right side of the fill
                    pos, entry_px, stop_px, entry_t = d, o[i], sp, t
        # --- intrabar stop ---
        if pos == 1 and l[i] <= stop_px:
            close_pos(min(o[i], stop_px), t, "stop")
        elif pos == -1 and h[i] >= stop_px:
            close_pos(max(o[i], stop_px), t, "stop")

    if pos != 0:
        close_pos(c[-1], idx[-1] + bar_len, "end_of_data")
    return trades
