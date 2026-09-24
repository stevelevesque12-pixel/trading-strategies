"""
Open-range volatility breakout (TR-scaled), intraday, on MNQ.

Rules (as supplied by the user):

  1. Entries. TR1 = true range of the previous full session.
       Long stop order:  Open + 0.25 x TR1
       Short stop order: Open - 0.25 x TR1
     "Open" is the 09:30 ET RTH open. Orders are active 09:30-13:00 ET.
  2. Stop. 0.25 x TR1 from entry = 1R (i.e. the stop sits at the open).
  3. Break-even. Once a *completed* bar has reached +0.50 x TR1 (+2R), the
     stop moves to entry starting from the next bar.
  4. Exit. Stop, break-even, or market exit at 15:55 ET. No target, no trail.
  5. Re-entry. After a stop-out, the same side re-arms once price trades
     back through its entry level from the other side. Max 3 entries per
     day across long + short.
  6. Sizing. Risk 1% of current (compounding) equity per trade, MNQ.
  7. Filters. None.
  Costs: $0.95 round-trip commission per contract + 1 tick slippage per
  side on every fill.

Bar-data modelling (the rules are tick-level; we only have OHLC bars):

  - Each bar is walked as a price path open -> nearer extreme -> farther
    extreme -> close, and consecutive bars are joined by a "gap" segment
    from the previous close to the next open. Stop orders fill at their
    level (+ slippage) when a normal segment crosses them, or at the gap
    price (+ slippage) when a gap jumps over them. This lets entries,
    stop-outs, re-arms and reverse-side entries happen in the right order
    inside a single bar, which matters on 15-minute data.
  - "+2R reached on a completed bar" uses the best price seen along the
    path *after* the fill, so the entry bar only counts from the fill on.
  - The 15:55 market exit fills at the close of the last bar that starts
    before 15:55 (exactly 15:55 on 1m data; 16:00 on 15m data -- a
    5-minute approximation).

TR1 session definition is configurable (`tr_session`):
  - "eth" (default): the full Globex day, 18:00 ET prior evening -> 17:00
    ET. This is what a futures daily bar is, and the natural reading of
    "previous full session".
  - "rth": 09:30-16:00 ET only.
"""

from dataclasses import dataclass, field
from datetime import time, timedelta
from math import floor
from typing import List, Literal, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class VolBreakoutConfig:
    entry_mult: float = 0.25
    stop_mult: float = 0.25
    breakeven_mult: float = 0.50
    session_open: time = time(9, 30)
    entries_until: time = time(13, 0)
    exit_at: time = time(15, 55)
    max_entries_per_day: int = 3
    risk_pct: float = 0.01
    commission_rt: float = 0.95  # dollars per contract, round trip
    slippage_ticks: int = 1  # per side, every fill
    tr_session: Literal["eth", "rth"] = "eth"
    tz: str = "America/New_York"


@dataclass
class VBTrade:
    date: object
    entry_time: object
    exit_time: object
    direction: str
    level: float  # the stop-order price (pre-slippage)
    entry_price: float  # actual fill, incl. slippage
    initial_stop: float
    exit_price: float  # actual fill, incl. slippage
    exit_reason: str  # stop | breakeven | eod
    contracts: int
    tr1: float
    risk_dollars: float  # planned 1R in dollars (level -> initial stop)
    pnl_dollars: float  # net of commission and slippage
    r_multiple: float  # pnl_dollars / risk_dollars
    equity_after: float


def true_ranges(df: pd.DataFrame, cfg: VolBreakoutConfig) -> pd.Series:
    """
    Per-session true range, indexed by session date. `df` must be ET-indexed
    OHLC. TR = max(high, prev_close) - min(low, prev_close).
    """
    if cfg.tr_session == "eth":
        # 18:00 ET belongs to the next day's session.
        key = (df.index + pd.Timedelta(hours=6)).date
        sess = df
    else:
        t = df.index.time
        mask = (t >= cfg.session_open) & (t < time(16, 0))
        sess = df[mask]
        key = sess.index.date
    agg = sess.groupby(key).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev_close = agg["close"].shift(1)
    hi = pd.concat([agg["high"], prev_close], axis=1).max(axis=1)
    lo = pd.concat([agg["low"], prev_close], axis=1).min(axis=1)
    return (hi - lo).rename("tr")


def _bar_path(o: float, h: float, l: float, c: float) -> Tuple[float, float, float, float]:
    # Nearer extreme first; ties go high-first.
    if abs(h - o) <= abs(o - l):
        return (o, h, l, c)
    return (o, l, h, c)


@dataclass
class _Position:
    direction: Literal["long", "short"]
    level: float
    fill: float
    stop: float
    contracts: int
    risk_dollars: float
    entry_time: object
    best: float  # most favourable price seen since the fill
    be_armed: bool = False  # +2R reached; stop moves to level after this bar


@dataclass
class _DayState:
    long_level: float
    short_level: float
    tr1: float
    equity: float
    entries: int = 0
    armed_long: bool = True
    armed_short: bool = True
    pos: Optional[_Position] = None
    trades: List[VBTrade] = field(default_factory=list)


def simulate_day(
    bars: Sequence[Tuple[object, float, float, float, float]],
    tr1: float,
    equity: float,
    tick_size: float,
    point_value: float,
    cfg: VolBreakoutConfig,
) -> Tuple[List[VBTrade], float]:
    """
    Run one RTH day. `bars` are (start_ts, open, high, low, close) from the
    09:30 bar onward, all starting before `cfg.exit_at`, time-ordered and
    ET-localized. Returns (trades, equity at end of day).
    """
    if not bars or tr1 <= 0:
        return [], equity

    day_open = bars[0][1]
    st = _DayState(
        long_level=day_open + cfg.entry_mult * tr1,
        short_level=day_open - cfg.entry_mult * tr1,
        tr1=tr1,
        equity=equity,
    )
    slip = cfg.slippage_ticks * tick_size
    date = bars[0][0].date()

    def close_pos(price_pre_slip: float, ts, reason: str) -> None:
        pos = st.pos
        sign = 1 if pos.direction == "long" else -1
        exit_fill = price_pre_slip - sign * slip
        pnl = (exit_fill - pos.fill) * sign * point_value * pos.contracts - cfg.commission_rt * pos.contracts
        st.equity += pnl
        st.trades.append(
            VBTrade(
                date=date,
                entry_time=pos.entry_time,
                exit_time=ts,
                direction=pos.direction,
                level=pos.level,
                entry_price=pos.fill,
                initial_stop=pos.level - sign * cfg.stop_mult * tr1,
                exit_price=exit_fill,
                exit_reason=reason,
                contracts=pos.contracts,
                tr1=tr1,
                risk_dollars=pos.risk_dollars,
                pnl_dollars=pnl,
                r_multiple=pnl / pos.risk_dollars if pos.risk_dollars else 0.0,
                equity_after=st.equity,
            )
        )
        st.pos = None

    def open_pos(direction: str, level: float, price_pre_slip: float, ts) -> None:
        st.entries += 1
        if direction == "long":
            st.armed_long = False
        else:
            st.armed_short = False
        risk_pts = cfg.stop_mult * tr1
        per_contract = risk_pts * point_value
        contracts = floor(st.equity * cfg.risk_pct / per_contract) if per_contract > 0 else 0
        if contracts < 1:
            return  # can't size even one contract at 1% risk; the order is spent
        sign = 1 if direction == "long" else -1
        fill = price_pre_slip + sign * slip
        st.pos = _Position(
            direction=direction,
            level=level,
            fill=fill,
            stop=level - sign * risk_pts,
            contracts=contracts,
            risk_dollars=contracts * per_contract,
            entry_time=ts,
            best=price_pre_slip,
        )

    def rearm(price: float) -> None:
        # Re-arm a side once price is back on the far side of its level.
        if price < st.long_level and (st.pos is None or st.pos.direction != "long"):
            st.armed_long = True
        if price > st.short_level and (st.pos is None or st.pos.direction != "short"):
            st.armed_short = True

    def walk(a: float, b: float, ts, orders_active: bool, gap: bool) -> None:
        """Process all events along the monotone segment a -> b, in order."""
        cur = a
        while True:
            rearm(cur)
            pos = st.pos
            if pos is not None:
                if pos.direction == "long":
                    pos.best = max(pos.best, cur)
                    if b <= pos.stop < cur or (cur == pos.stop and b < cur):
                        px = b if gap else pos.stop
                        reason = "breakeven" if pos.stop == pos.level else "stop"
                        close_pos(px, ts, reason)
                        cur = px
                        continue
                    pos.best = max(pos.best, b)
                else:
                    pos.best = min(pos.best, cur)
                    if cur < pos.stop <= b or (cur == pos.stop and b > cur):
                        px = b if gap else pos.stop
                        reason = "breakeven" if pos.stop == pos.level else "stop"
                        close_pos(px, ts, reason)
                        cur = px
                        continue
                    pos.best = min(pos.best, b)
                return

            if not orders_active or st.entries >= cfg.max_entries_per_day:
                return
            if b > cur and st.armed_long and cur < st.long_level <= b:
                px = b if gap else st.long_level
                open_pos("long", st.long_level, px, ts)
                cur = px
                continue
            if b < cur and st.armed_short and b <= st.short_level < cur:
                px = b if gap else st.short_level
                open_pos("short", st.short_level, px, ts)
                cur = px
                continue
            return

    prev_close: Optional[float] = None
    for ts, o, h, l, c in bars:
        active = ts.time() < cfg.entries_until
        if prev_close is not None and o != prev_close:
            walk(prev_close, o, ts, active, gap=True)
        path = _bar_path(o, h, l, c)
        for a, b in zip(path, path[1:]):
            if a != b:
                walk(a, b, ts, active, gap=False)

        # Bar complete: apply break-even for the next bar if +2R was reached.
        pos = st.pos
        if pos is not None and pos.stop != pos.level:
            sign = 1 if pos.direction == "long" else -1
            if (pos.best - pos.level) * sign >= cfg.breakeven_mult * tr1:
                pos.stop = pos.level
        prev_close = c

    if st.pos is not None:
        close_pos(prev_close, bars[-1][0] + _bar_len(bars), "eod")

    return st.trades, st.equity


def _bar_len(bars) -> timedelta:
    if len(bars) >= 2:
        return bars[1][0] - bars[0][0]
    return timedelta(0)


def run_backtest(
    df: pd.DataFrame,
    tick_size: float,
    point_value: float,
    start_equity: float = 100_000.0,
    cfg: Optional[VolBreakoutConfig] = None,
    start_date=None,
    end_date=None,
) -> List[VBTrade]:
    """
    `df`: ET-localized OHLC(V) bars at any intraday resolution that divides
    09:30 (1m/5m/15m). Returns every trade with compounding equity.
    Optional `start_date`/`end_date` (datetime.date, inclusive) limit which
    days are traded; bars before the window still feed TR1.
    """
    cfg = cfg or VolBreakoutConfig()
    trs = true_ranges(df, cfg)
    tr_dates = list(trs.index)
    tr_pos = {d: i for i, d in enumerate(tr_dates)}

    t = df.index.time
    rth = df[(t >= cfg.session_open) & (t < cfg.exit_at)]

    equity = start_equity
    trades: List[VBTrade] = []
    for d, day in rth.groupby(rth.index.date):
        if (start_date and d < start_date) or (end_date and d > end_date):
            continue
        if day.index[0].time() != cfg.session_open:
            continue  # no 09:30 bar -> no defined open
        # Previous full session: the last session strictly before today.
        # (For ETH, today's session key is d itself -- 18:00 on d-1 onward.)
        i = tr_pos.get(d)
        if i is None or i == 0:
            continue
        tr1 = trs.iloc[i - 1]
        if pd.isna(tr1):
            continue
        bars = list(zip(day.index, day["open"], day["high"], day["low"], day["close"]))
        day_trades, equity = simulate_day(bars, float(tr1), equity, tick_size, point_value, cfg)
        trades.extend(day_trades)
    return trades
