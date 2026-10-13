"""
NQ Keltner Pulse (M15): Keltner Channel breakout with an ATR-buffered pending
STOP entry.

Rules (long case; short is the exact mirror):

  1. Keltner Channel: middle = EMA(close, kc_period), band = middle +/-
     kc_mult * EMA(true range, kc_period). This matches TradingView's
     `ta.kc(close, 20, 2.6)` (useTrueRange defaults to true).
  2. Signal: a bar *closes* above the upper band. At that bar's close, place a
     buy STOP at signal_bar.high + entry_buffer_atr * ATR. (ATR is Wilder's,
     `atr_period` bars, i.e. `ta.atr`.) Stop/target distances are frozen at
     signal time using that same ATR.
  3. The pending order expires `pending_expiry_bars` bars after it was placed.
     A fresh signal while an order is pending replaces it (same or opposite
     direction). No orders are placed while a position is open.
  4. Fill: the first later bar whose high reaches the stop price. If that bar
     gaps through it, the fill is at the bar's open (realistic slippage).
  5. Exits, checked in this order on every bar after the fill:
       - Stop-loss: entry - stop_atr * ATR, ratcheted up by the trailing stop.
       - Profit target: entry * (1 + profit_target_pct / 100).
       - Time exit: at the close of the `time_exit_bars`-th bar held
         (counting the fill bar as bar 1).
     A bar touching both stop and target is booked as the stop (no intrabar
     data, so assume the worse outcome). Stops/targets gapped through fill at
     the open.
  6. Trailing stop: once the highest high since entry is >= entry +
     trail_activation_atr * ATR, the stop trails trail_points behind that
     highest high (never loosened). The trail is updated from a bar's high
     *after* that bar's exits are checked, so a bar can't both set a new high
     and be stopped by the trail it just created.

Unspecified in the source spec, chosen here (all overridable in
KeltnerPulseConfig): the ATR period for buffer/stop/activation (14), the
signal-bar high/low as the breakout reference, and replace-on-new-signal.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from backtest.engine import Trade
from failed2s.instruments import Instrument


@dataclass
class KeltnerPulseConfig:
    kc_period: int = 20
    kc_mult: float = 2.6
    atr_period: int = 14
    entry_buffer_atr: float = 0.1
    stop_atr: float = 2.5
    profit_target_pct: float = 1.5
    trail_points: float = 100.0
    trail_activation_atr: float = 1.8
    time_exit_bars: int = 30
    pending_expiry_bars: int = 46
    # Costs: round-turn commission per contract (dollars), and adverse
    # slippage in ticks applied to every stop-type fill (entry, stop-loss,
    # trailing stop). Targets (limits) and time exits get no slippage.
    commission_rt: float = 0.0
    slippage_ticks: float = 0.0


def add_indicators(df: pd.DataFrame, cfg: KeltnerPulseConfig) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [out["high"] - out["low"], (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = out["high"].iloc[0] - out["low"].iloc[0]

    mid = out["close"].ewm(span=cfg.kc_period, adjust=False).mean()
    rng = tr.ewm(span=cfg.kc_period, adjust=False).mean()
    out["kc_mid"] = mid
    out["kc_upper"] = mid + cfg.kc_mult * rng
    out["kc_lower"] = mid - cfg.kc_mult * rng
    out["atr"] = tr.ewm(alpha=1.0 / cfg.atr_period, adjust=False).mean()

    # Indicators are unreliable until enough history has accumulated.
    warmup = max(cfg.kc_period, cfg.atr_period)
    out.iloc[:warmup, out.columns.get_indexer(["kc_mid", "kc_upper", "kc_lower", "atr"])] = np.nan
    return out


@dataclass
class _Pending:
    direction: Literal["long", "short"]
    price: float
    stop_dist: float
    trail_activation_dist: float
    placed_idx: int


@dataclass
class _Position:
    direction: Literal["long", "short"]
    entry_price: float
    entry_time: object
    entry_idx: int
    stop_price: float
    initial_stop: float
    target_price: float
    trail_activation_dist: float
    best: float  # highest high (long) / lowest low (short) since entry


@dataclass
class KeltnerPulseBacktest:
    instrument: Instrument
    cfg: KeltnerPulseConfig = field(default_factory=KeltnerPulseConfig)
    contracts: int = 1

    def run(self, df: pd.DataFrame) -> List[Trade]:
        cfg = self.cfg
        d = add_indicators(df, cfg)
        o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        up, lo, atr = d["kc_upper"].to_numpy(), d["kc_lower"].to_numpy(), d["atr"].to_numpy()
        idx = d.index
        slip = cfg.slippage_ticks * self.instrument.tick_size

        trades: List[Trade] = []
        pending: Optional[_Pending] = None
        pos: Optional[_Position] = None

        def close(i: int, price: float, reason: str) -> None:
            nonlocal pos
            sign = 1 if pos.direction == "long" else -1
            pnl_points = (price - pos.entry_price) * sign
            pnl_dollars = pnl_points * self.instrument.point_value * self.contracts - cfg.commission_rt * self.contracts
            risk = abs(pos.entry_price - pos.initial_stop)
            trades.append(
                Trade(
                    entry_time=pos.entry_time,
                    exit_time=idx[i],
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    stop_price=pos.initial_stop,
                    target_price=pos.target_price,
                    exit_price=price,
                    exit_reason=reason,
                    contracts=self.contracts,
                    pnl_points=pnl_points,
                    pnl_dollars=pnl_dollars,
                    r_multiple=pnl_points / risk if risk else 0.0,
                )
            )
            pos = None

        for i in range(len(d)):
            # 1. Pending order: expire, else try to fill on this bar.
            if pending is not None and pos is None:
                if i - pending.placed_idx > cfg.pending_expiry_bars:
                    pending = None
                elif pending.direction == "long" and h[i] >= pending.price:
                    fill = max(o[i], pending.price) + slip
                    pos = self._open(pending, fill, idx[i], i)
                    pending = None
                elif pending.direction == "short" and l[i] <= pending.price:
                    fill = min(o[i], pending.price) - slip
                    pos = self._open(pending, fill, idx[i], i)
                    pending = None

            # 2. Manage the open position (including on its fill bar).
            if pos is not None:
                if pos.direction == "long":
                    if l[i] <= pos.stop_price:
                        reason = "trail_stop" if pos.stop_price > pos.initial_stop else "stop"
                        close(i, min(o[i], pos.stop_price) - slip, reason)
                    elif h[i] >= pos.target_price:
                        close(i, max(o[i], pos.target_price) if i > pos.entry_idx else pos.target_price, "target")
                else:
                    if h[i] >= pos.stop_price:
                        reason = "trail_stop" if pos.stop_price < pos.initial_stop else "stop"
                        close(i, max(o[i], pos.stop_price) + slip, reason)
                    elif l[i] <= pos.target_price:
                        close(i, min(o[i], pos.target_price) if i > pos.entry_idx else pos.target_price, "target")

                if pos is not None and i - pos.entry_idx + 1 >= cfg.time_exit_bars:
                    close(i, c[i], "time")

                if pos is not None:
                    self._update_trail(pos, h[i], l[i])

            # 3. New signal at this bar's close (only while flat).
            if pos is None and not np.isnan(atr[i]):
                direction = None
                if c[i] > up[i]:
                    direction, price = "long", h[i] + cfg.entry_buffer_atr * atr[i]
                elif c[i] < lo[i]:
                    direction, price = "short", l[i] - cfg.entry_buffer_atr * atr[i]
                if direction is not None:
                    price = self._round_tick(price)
                    pending = _Pending(direction, price, cfg.stop_atr * atr[i], cfg.trail_activation_atr * atr[i], i)

        if pos is not None:
            close(len(d) - 1, c[-1], "end_of_data")
        return trades

    def _open(self, p: _Pending, fill: float, ts, i: int) -> _Position:
        sign = 1 if p.direction == "long" else -1
        stop = self._round_tick(fill - sign * p.stop_dist)
        target = self._round_tick(fill * (1 + sign * self.cfg.profit_target_pct / 100.0))
        return _Position(p.direction, fill, ts, i, stop, stop, target, p.trail_activation_dist, fill)

    def _update_trail(self, pos: _Position, high: float, low: float) -> None:
        if pos.direction == "long":
            pos.best = max(pos.best, high)
            if pos.best - pos.entry_price >= pos.trail_activation_dist:
                pos.stop_price = max(pos.stop_price, pos.best - self.cfg.trail_points)
        else:
            pos.best = min(pos.best, low)
            if pos.entry_price - pos.best >= pos.trail_activation_dist:
                pos.stop_price = min(pos.stop_price, pos.best + self.cfg.trail_points)

    def _round_tick(self, price: float) -> float:
        t = self.instrument.tick_size
        return round(round(price / t) * t, 10)
