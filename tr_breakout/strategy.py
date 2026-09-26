"""
TR1 breakout: stop-entry orders at RTH open +/- 0.25 x prior-session true range.

Rules (as specified):
1. TR1 = true range of the previous full session. Long stop-entry at
   Open + k*TR1, short stop-entry at Open - k*TR1 (k = 0.25). Orders active
   09:30-13:00 ET.
2. Stop = k*TR1 from entry = 1R.
3. Break-even: once a *completed* bar has traded to +2k*TR1 (= +2R) beyond
   entry, the stop moves to entry starting with the next bar.
4. Exit: stop, break-even, or market exit at 15:55 ET. No target/trailing.
5. Re-entry after a stop-out once price trades back through the entry level
   from the other side. Max 3 entries per day across long + short.
6. Sizing: risk 1% of the $100k starting equity ($1,000) per trade in MNQ,
   fixed -- no compounding (`compound=True` sizes off current equity instead).
7. No filters.
Costs: $0.95 round-trip commission per contract + 1 tick slippage per side.

Interpretation choices (the spec is silent on these; all are parameters):
- "Session" = RTH 09:30-16:00 ET (`session="rth"`), "Open" = the 09:30 RTH
  open. `session="eth"` uses the full 18:00-17:00 Globex day for TR1 instead.
- TR1 = max(H, prev close) - min(L, prev close) of the prior session.
- Entry/stop levels are rounded to the instrument tick.
- Intrabar path is modeled as O->L->H->C for up bars and O->H->L->C for down
  bars, so entries, stop-outs and re-entries inside one bar resolve in a
  consistent (if assumed) order. Stops that gap through fill at the bar open.
- Break-even trigger uses the favorable extreme reached *after* the fill
  within the path (the entry bar counts as a completed bar once it closes).
- Market exit at 15:55 fills at the open of the first bar starting >= 15:55;
  on bars too coarse to have one (e.g. 15m), at the close of the last bar
  starting before 15:55 (i.e. 16:00 on 15m data).
- Contracts = floor(1% x sizing equity / (stop points x point value)); a day whose
  stop is so wide that 0 contracts fit is skipped.
"""

from dataclasses import dataclass, field
from datetime import date, time
from math import floor
from typing import List, Optional

import pandas as pd


@dataclass
class Config:
    k: float = 0.25                 # entry offset and stop, as a fraction of TR1
    be_mult: float = 2.0            # break-even trigger, in R
    entry_start: time = time(9, 30)
    entry_end: time = time(13, 0)
    rth_open: time = time(9, 30)
    rth_close: time = time(16, 0)
    exit_at: time = time(15, 55)
    session: str = "rth"            # "rth" or "eth" for the TR1 session definition
    max_entries: int = 3
    risk_pct: float = 0.01
    start_equity: float = 100_000.0
    fixed_contracts: Optional[int] = None  # trade exactly N contracts, ignoring risk_pct
    max_contracts: Optional[int] = None  # position cap (prop-firm limits)
    compound: bool = False          # False: always risk risk_pct of start_equity
    tick_size: float = 0.25
    point_value: float = 2.0        # MNQ
    commission_rt: float = 0.95     # per contract, round trip
    slippage_ticks: int = 1         # per side


@dataclass
class Trade:
    day: date
    side: str
    entry_time: pd.Timestamp
    entry_level: float
    entry_price: float
    stop_price: float
    contracts: int
    tr1: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    be_moved: bool = False
    pnl: float = 0.0
    r_multiple: float = 0.0
    equity_after: float = 0.0


def _round_tick(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 10)


def session_true_ranges(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """TR per session, indexed by the RTH calendar date it belongs to."""
    if cfg.session == "rth":
        t = df.index.time
        s = df[(t >= cfg.rth_open) & (t < cfg.rth_close)]
        key = s.index.date
    elif cfg.session == "eth":
        # Globex day starting 18:00 ET belongs to the next calendar date.
        s = df
        key = (s.index + pd.Timedelta(hours=6)).date
    else:
        raise ValueError(f"unknown session {cfg.session!r}")
    g = s.groupby(key).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev_close = g["close"].shift()
    tr = pd.concat([g["high"], prev_close], axis=1).max(axis=1) - pd.concat(
        [g["low"], prev_close], axis=1
    ).min(axis=1)
    return tr


def _path(o: float, h: float, l: float, c: float) -> List[float]:
    return [o, l, h, c] if c >= o else [o, h, l, c]


class _Day:
    """State machine for one RTH session."""

    def __init__(self, day, open_px, tr1, cfg: Config, equity: float):
        self.cfg = cfg
        self.day = day
        self.tr1 = tr1
        self.dist = _round_tick(cfg.k * tr1, cfg.tick_size)
        self.long_lvl = _round_tick(open_px + cfg.k * tr1, cfg.tick_size)
        self.short_lvl = _round_tick(open_px - cfg.k * tr1, cfg.tick_size)
        # An order is armed while price sits on the far side of its level.
        self.armed = {"long": open_px < self.long_lvl, "short": open_px > self.short_lvl}
        self.entries = 0
        self.unsizable = self.dist <= 0
        self.pos: Optional[Trade] = None
        self.best = 0.0            # favorable extreme since entry
        self.be_pending = False    # BE earned this bar, applies from next bar
        self.equity = equity
        self.trades: List[Trade] = []
        self.slip = cfg.slippage_ticks * cfg.tick_size

    # --- helpers -----------------------------------------------------------
    def _open(self, side, ts, trigger_px):
        cfg = self.cfg
        sizing_equity = self.equity if cfg.compound else cfg.start_equity
        contracts = floor(cfg.risk_pct * sizing_equity / (self.dist * cfg.point_value))
        if cfg.fixed_contracts is not None:
            contracts = cfg.fixed_contracts
        if cfg.max_contracts is not None:
            contracts = min(contracts, cfg.max_contracts)
        if contracts < 1:
            self.unsizable = True  # stop too wide for 1% risk: stand aside today
            return
        self.entries += 1
        lvl = self.long_lvl if side == "long" else self.short_lvl
        sgn = 1 if side == "long" else -1
        fill = trigger_px + sgn * self.slip
        stop = lvl - sgn * self.dist
        self.pos = Trade(self.day, side, ts, lvl, fill, stop, contracts, self.tr1)
        self.best = trigger_px
        self.armed[side] = False

    def _close(self, ts, trigger_px, reason):
        cfg, p = self.cfg, self.pos
        sgn = 1 if p.side == "long" else -1
        p.exit_time, p.exit_reason = ts, reason
        p.exit_price = trigger_px - sgn * self.slip
        pts = sgn * (p.exit_price - p.entry_price)
        p.pnl = pts * cfg.point_value * p.contracts - cfg.commission_rt * p.contracts
        p.r_multiple = p.pnl / (self.dist * cfg.point_value * p.contracts)
        self.equity += p.pnl
        p.equity_after = self.equity
        self.trades.append(p)
        self.pos = None
        self.be_pending = False

    # --- bar processing ----------------------------------------------------
    def on_bar(self, ts, o, h, l, c, entries_open: bool):
        if self.pos is not None and self.be_pending and not self.pos.be_moved:
            self.pos.stop_price = self.pos.entry_level
            self.pos.be_moved = True
        self.be_pending = False

        pts = _path(o, h, l, c)
        # Gap at the bar open: price jumps straight to `o`.
        self._walk(ts, pts[0], pts[0], entries_open, gap=True)
        for a, b in zip(pts, pts[1:]):
            self._walk(ts, a, b, entries_open, gap=False)

        p = self.pos
        if p is not None and not p.be_moved:
            sgn = 1 if p.side == "long" else -1
            if sgn * (self.best - p.entry_level) >= self.cfg.be_mult * self.dist - 1e-9:
                self.be_pending = True

    def _walk(self, ts, a, b, entries_open, gap):
        """Move price from a to b, firing every level crossed in order."""
        cur = a
        while True:
            events = []
            up = b >= cur
            if self.pos is not None:
                p = self.pos
                if p.side == "long" and ((gap and b <= p.stop_price) or (not up and b <= p.stop_price <= cur)):
                    events.append((abs(cur - p.stop_price), "stop", p.stop_price))
                if p.side == "short" and ((gap and b >= p.stop_price) or (up and cur <= p.stop_price <= b)):
                    events.append((abs(cur - p.stop_price), "stop", p.stop_price))
            elif entries_open and self.entries < self.cfg.max_entries and not self.unsizable:
                if self.armed["long"] and ((gap and b >= self.long_lvl) or (up and cur <= self.long_lvl <= b)):
                    events.append((abs(cur - self.long_lvl), "long", self.long_lvl))
                if self.armed["short"] and ((gap and b <= self.short_lvl) or (not up and b <= self.short_lvl <= cur)):
                    events.append((abs(cur - self.short_lvl), "short", self.short_lvl))
            # Re-arm orders once price trades back to the far side of their level.
            lo, hi = min(cur, b), max(cur, b)
            if lo < self.long_lvl and not self.armed["long"] and self._may_rearm("long"):
                self.armed["long"] = True
            if hi > self.short_lvl and not self.armed["short"] and self._may_rearm("short"):
                self.armed["short"] = True

            if not events:
                if self.pos is not None:
                    self.best = max(self.best, b) if self.pos.side == "long" else min(self.best, b)
                return
            events.sort(key=lambda e: e[0])
            _, kind, lvl = events[0]
            px = b if gap else lvl  # gapped through: fill at the open
            if kind == "stop":
                p = self.pos
                if p.side == "long":
                    self.best = max(self.best, px)
                else:
                    self.best = min(self.best, px)
                self._close(ts, px, "breakeven" if p.be_moved else "stop")
            else:
                self._open(kind, ts, px)
            if gap:
                return
            cur = lvl
            # Price sits exactly on a level now; a re-entry needs it to leave
            # to the far side first, which `lo < level` handles next pass.

    def _may_rearm(self, side):
        # Can't re-arm the side we're currently holding.
        return self.pos is None or self.pos.side != side

    def flatten(self, ts, px):
        if self.pos is not None:
            self._close(ts, px, "time_exit")


def run_backtest(df: pd.DataFrame, cfg: Optional[Config] = None) -> List[Trade]:
    """`df`: tz-aware (America/New_York) OHLC bars, any intraday resolution."""
    cfg = cfg or Config()
    tr = session_true_ranges(df, cfg)
    t = df.index.time
    rth = df[(t >= cfg.rth_open) & (t < cfg.rth_close)]
    equity = cfg.start_equity
    trades: List[Trade] = []

    days = list(rth.groupby(rth.index.date))
    tr_dates = list(tr.index)
    tr_pos = {d: i for i, d in enumerate(tr_dates)}
    for day, bars in days:
        if bars.index[0].time() != cfg.rth_open:
            continue  # no 09:30 bar -> no clean open (half days, gaps)
        # Previous session's TR: RTH -> prior RTH date; ETH -> the Globex
        # session that ended the evening before today's RTH (same key - 1).
        i = tr_pos.get(day)
        if i is None or i == 0:
            continue
        tr1 = tr.iloc[i - 1]
        if not tr1 > 0:
            continue
        state = _Day(day, bars["open"].iloc[0], tr1, cfg, equity)
        flattened = False
        for ts, row in bars.iterrows():
            bt = ts.time()
            if bt >= cfg.exit_at:
                state.flatten(ts, row["open"])
                flattened = True
                break
            state.on_bar(ts, row["open"], row["high"], row["low"], row["close"],
                         cfg.entry_start <= bt < cfg.entry_end)
        if not flattened:
            state.flatten(bars.index[-1], bars["close"].iloc[-1])
        trades.extend(state.trades)
        equity = state.equity
    return trades


def summarize(trades: List[Trade], cfg: Config) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = pd.Series([t.pnl for t in trades])
    eq = pd.Series([cfg.start_equity] + [t.equity_after for t in trades])
    dd = (eq / eq.cummax() - 1).min()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    out = {
        "trades": len(trades),
        "win_rate_%": 100 * len(wins) / len(pnl),
        "avg_R": sum(t.r_multiple for t in trades) / len(trades),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() else float("inf"),
        "net_pnl": pnl.sum(),
        "end_equity": eq.iloc[-1],
        "return_%": 100 * (eq.iloc[-1] / cfg.start_equity - 1),
        "max_dd_%": 100 * dd,
        "max_dd_$": (eq - eq.cummax()).min(),
        "costs": sum(t.contracts * (cfg.commission_rt + 2 * cfg.slippage_ticks * cfg.tick_size * cfg.point_value)
                     for t in trades),
    }
    return {k: v if k == "trades" else round(float(v), 3 if k == "avg_R" else 2 if k == "profit_factor" else 1) for k, v in out.items()}


def trades_frame(trades: List[Trade]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in trades])
