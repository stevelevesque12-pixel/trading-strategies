"""
Backtester + CLI for the SuperTrend/ATR strategy.

Usage:
    python -m supertrend_atr.backtest --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet \
        --symbol NQ --tf 15min --session rth
    python -m supertrend_atr.backtest --data ... --symbol NQ --compare 5min,15min,30min,1h

Fill model (bar OHLC only, no intrabar path):
  - A pending buy stop placed at bar N's close can fill from bar N+1 on:
    at the open if the bar gaps through it, otherwise at the stop price.
    Sell stops mirror. `slippage_ticks` is added against us on every stop
    fill (entries and stop/trail exits).
  - Protective stops fill the same way (gap -> open, else stop price).
  - On the bar an entry fills, the initial stop is also checked against
    that bar's range -- if both are inside the range the stop is assumed
    hit (conservative; the order of the touches is unknown).
  - The trailing stop is recomputed at each bar's close and takes effect
    from the next bar, so it never uses a high/low it couldn't have seen.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from backtest.data import load_1m_csv, resample_ohlc
from backtest.engine import Trade
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS, Instrument
from failed2s.strategy import SessionConfig

from .strategy import PendingStop, Position, StrategyParams, SuperTrendATRStrategy


class SuperTrendBacktester:
    def __init__(
        self,
        instrument: Instrument,
        params: Optional[StrategyParams] = None,
        timeframe: str = "15min",
        session_mode: str = "rth",
        session: Optional[SessionConfig] = None,
        contracts: int = 1,
        slippage_ticks: int = 1,
        commission_rt: float = 0.0,
    ):
        if session_mode not in ("rth", "full"):
            raise ValueError("session_mode must be 'rth' or 'full'")
        self.instrument = instrument
        self.params = params or StrategyParams()
        self.timeframe = timeframe
        self.session_mode = session_mode
        self.session = session or SessionConfig()
        self.contracts = contracts
        self.slip = slippage_ticks * instrument.tick_size
        self.commission_rt = commission_rt

    # -- helpers ----------------------------------------------------------

    def _can_enter(self, ts) -> bool:
        if self.session_mode == "full":
            return True
        t = ts.time()
        return self.session.session_start <= t < self.session.no_entry_after

    def _close(self, pos: Position, price: float, ts, reason: str) -> Trade:
        sign = 1 if pos.direction == "long" else -1
        pts = (price - pos.entry_price) * sign
        risk = abs(pos.entry_price - pos.initial_stop)
        dollars = pts * self.instrument.point_value * self.contracts - self.commission_rt * self.contracts
        return Trade(
            entry_time=pos.entry_time, exit_time=ts, direction=pos.direction,
            entry_price=pos.entry_price, stop_price=pos.initial_stop, target_price=float("nan"),
            exit_price=price, exit_reason=reason, contracts=self.contracts,
            pnl_points=pts, pnl_dollars=dollars, r_multiple=pts / risk if risk else 0.0,
        )

    def _stop_fill(self, pos: Position, bar: Bar) -> Optional[float]:
        """Exit price if the protective stop is hit on this bar, else None."""
        if pos.direction == "long":
            if bar.open <= pos.stop:
                return bar.open - self.slip
            if bar.low <= pos.stop:
                return pos.stop - self.slip
        else:
            if bar.open >= pos.stop:
                return bar.open + self.slip
            if bar.high >= pos.stop:
                return pos.stop + self.slip
        return None

    def _entry_fill(self, order: PendingStop, bar: Bar) -> Optional[float]:
        if order.direction == "long":
            if bar.open >= order.price:
                return bar.open + self.slip
            if bar.high >= order.price:
                return order.price + self.slip
        else:
            if bar.open <= order.price:
                return bar.open - self.slip
            if bar.low <= order.price:
                return order.price - self.slip
        return None

    # -- main loop --------------------------------------------------------

    def run(self, data) -> List[Trade]:
        base = data if isinstance(data, pd.DataFrame) else load_1m_csv(data, tz=self.session.tz)
        df = resample_ohlc(base, self.timeframe)
        tf_len = pd.Timedelta(self.timeframe)

        strat = SuperTrendATRStrategy(self.params, tick_size=self.instrument.tick_size)

        htf_rows, htf_close = [], []
        if self.params.htf_multiple > 0:
            htf_len = tf_len * self.params.htf_multiple
            htf_df = resample_ohlc(base, htf_len)
            htf_rows = list(htf_df.itertuples(index=True))
            htf_close = (htf_df.index + htf_len).to_numpy()
        hptr = 0

        trades: List[Trade] = []
        pos: Optional[Position] = None
        order: Optional[PendingStop] = None
        last = None

        for row in df.itertuples(index=True):
            ts = row.Index
            bar = Bar(ts, row.open, row.high, row.low, row.close, row.volume)
            bar_close_time = ts + tf_len

            # Safety net for rth mode: never hold across a session-date change
            # (a data gap that skipped the flatten bar).
            if pos is not None and self.session_mode == "rth" and ts.date() != pos.entry_time.date():
                trades.append(self._close(pos, bar.open, ts, "session_flatten"))
                pos = None

            # 1. Protective stop on an existing position.
            if pos is not None:
                px = self._stop_fill(pos, bar)
                if px is not None:
                    trades.append(self._close(pos, px, ts, "trail" if pos.stop != pos.initial_stop else "stop"))
                    pos = None

            # 2. Pending stop entry.
            if order is not None and pos is None:
                if not self._can_enter(ts):
                    order = None
                else:
                    fill = self._entry_fill(order, bar)
                    if fill is not None:
                        init = order.initial_stop(fill, self.params.sl_atr_mult)
                        best = bar.high if order.direction == "long" else bar.low
                        pos = Position(order.direction, fill, ts, init, init, best)
                        order = None
                        # Fill is at/after the open, so the open is never beyond
                        # the stop here -- this only catches an in-range touch.
                        px = self._stop_fill(pos, bar)
                        if px is not None:
                            trades.append(self._close(pos, px, ts, "stop"))
                            pos = None

            # 3. Close of this bar: feed indicators (HTF first, closed bars only).
            while hptr < len(htf_rows) and htf_close[hptr] <= bar_close_time:
                h = htf_rows[hptr]
                strat.on_htf_bar(Bar(h.Index, h.open, h.high, h.low, h.close, h.volume))
                hptr += 1
            strat.on_bar(bar)

            # 4. Manage the pending order and position with this bar's info.
            if order is not None:
                order.bars_left -= 1
                if order.bars_left <= 0 or strat.should_cancel(order):
                    order = None

            if pos is not None:
                strat.update_trailing(pos, bar)
                if self.session_mode == "rth" and bar_close_time.time() >= self.session.flatten_at:
                    trades.append(self._close(pos, bar.close, bar_close_time, "session_flatten"))
                    pos = None

            if pos is None and order is None and self._can_enter(ts):
                order = strat.entry_signal(bar)

            last = (bar_close_time, bar.close)

        if pos is not None and last is not None:
            trades.append(self._close(pos, last[1], last[0], "end_of_data"))
        return trades


def summarize(trades: List[Trade]) -> dict:
    m = compute_metrics(trades)
    if trades:
        m["exits"] = dict(Counter(t.exit_reason for t in trades))
        m["long_pnl"] = round(sum(t.pnl_dollars for t in trades if t.direction == "long"), 2)
        m["short_pnl"] = round(sum(t.pnl_dollars for t in trades if t.direction == "short"), 2)
        held = [(pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 60 for t in trades]
        m["avg_minutes_held"] = round(sum(held) / len(held), 1)
    return m


def main() -> None:
    d = StrategyParams()
    ap = argparse.ArgumentParser(description="Backtest the SuperTrend/ATR strategy")
    ap.add_argument("--data", required=True, help="OHLCV CSV or parquet (resolution <= --tf)")
    ap.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    ap.add_argument("--tf", default="15min", help="Entry timeframe, e.g. 5min, 15min, 1h")
    ap.add_argument("--compare", default=None, help="Comma-separated timeframes to run side by side")
    ap.add_argument("--session", default="rth", choices=["rth", "full"],
                    help="rth: enter 9:30-15:45 ET, flatten 15:55. full: trade around the clock, hold overnight")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--slippage-ticks", type=int, default=1)
    ap.add_argument("--commission-rt", type=float, default=4.0, help="Round-trip $ per contract")
    ap.add_argument("--st-atr-period", type=int, default=d.st_atr_period)
    ap.add_argument("--st-factor", type=float, default=d.st_factor)
    ap.add_argument("--atr-period", type=int, default=d.atr_period)
    ap.add_argument("--sl-mult", type=float, default=d.sl_atr_mult)
    ap.add_argument("--trail-mult", type=float, default=d.trail_atr_mult)
    ap.add_argument("--trail-activation", type=float, default=None,
                    help="Profit (in ATRs) before the trail engages; default = --trail-mult, 0 = trail immediately")
    ap.add_argument("--order-expiry", type=int, default=d.order_expiry_bars)
    ap.add_argument("--htf-multiple", type=int, default=d.htf_multiple)
    ap.add_argument("--vol-lookback", type=int, default=d.vol_lookback)
    ap.add_argument("--vol-min", type=float, default=d.vol_min)
    ap.add_argument("--vol-max", type=float, default=d.vol_max)
    ap.add_argument("--out", default="supertrend_trades.csv")
    a = ap.parse_args()

    params = StrategyParams(
        st_atr_period=a.st_atr_period, st_factor=a.st_factor, atr_period=a.atr_period,
        sl_atr_mult=a.sl_mult, trail_atr_mult=a.trail_mult, trail_activation_mult=a.trail_activation,
        order_expiry_bars=a.order_expiry, htf_multiple=a.htf_multiple,
        vol_lookback=a.vol_lookback, vol_min=a.vol_min, vol_max=a.vol_max,
    )
    print("Params:", asdict(params))
    base = load_1m_csv(a.data)
    tfs = a.compare.split(",") if a.compare else [a.tf]
    for tf in tfs:
        bt = SuperTrendBacktester(
            INSTRUMENTS[a.symbol], params, timeframe=tf, session_mode=a.session,
            contracts=a.contracts, slippage_ticks=a.slippage_ticks, commission_rt=a.commission_rt,
        )
        trades = bt.run(base)
        out = a.out if len(tfs) == 1 else a.out.replace(".csv", f"_{tf}.csv")
        write_trades_csv(trades, out)
        print(f"\n== {a.symbol} {tf} session={a.session} -> {out}")
        for k, v in summarize(trades).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
