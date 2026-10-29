"""
CrudePut30 -- sell a ~30-delta monthly WTI put, ported from the user's
platform strategy so it can be backtested on the futures bars in
`sample_data/real_multi_instrument/`.

Rules (mirroring the original `CrudePut30` class):
  * Entry: in the bar starting at `entry_time` (10:00), when flat, if the
    front monthly expiry is a new cycle and >= `min_dte_entry` (15)
    calendar days away -> sell 1 put at the strike with |delta| nearest 0.30.
    One entry per expiry cycle (the cycle is consumed even if we exit early).
  * Take profit: from `entry_time` onward, buy back once the put's mark is
    <= 50% of the credit.
  * Stop: off by default, as in the original (`elif False and ...`).
    `stop_mult=3.0` turns on the "premium 3x credit" stop it was written for.
  * Time exit: from `exit_time` on a day with DTE <= 1, flatten
    ("one session before option expiry").

No option prices exist in this repo, so every mark is a Black-76 model
price. Implied vol = trailing realized vol * `iv_mult`, fixed at the entry
value, or (by default) re-estimated daily so vol spikes in a selloff show up
in the buy-back price. Realized vol understates true IV most of the time
(the variance risk premium this strategy harvests) and the put skew isn't
modeled, so treat `iv_mult` as the key assumption and look at a range of it.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from . import black76
from .expiry import expiries_between, front_expiry

EXPIRY_TIME = time(14, 30)  # WTI options settle with the 14:30 ET futures settlement


@dataclass
class Put30Config:
    target_delta: float = 0.30
    min_dte_entry: int = 15
    take_profit_frac: float = 0.5        # buy back at this fraction of credit
    stop_mult: Optional[float] = None    # e.g. 3.0 -> exit when mark >= 3x credit
    entry_time: time = time(10, 0)
    exit_time: time = time(14, 0)        # on DTE <= 1 days
    multiplier: float = 1000.0           # bbl per contract: LO 1000, Micro WTI options 100
    strike_tick: float = 0.5
    iv_mult: float = 1.0
    rv_lookback_days: int = 20
    sticky_iv: bool = False              # True = hold entry IV for the whole trade
    slippage: float = 0.02               # USD/bbl per side
    commission: float = 2.5              # USD per contract per side
    lots: int = 1


@dataclass
class OptionTrade:
    entry_time: datetime
    exit_time: datetime
    expiry: date
    strike: float
    entry_fut: float
    exit_fut: float
    entry_iv: float
    credit: float
    exit_price: float
    reason: str
    pnl_dollars: float
    r_multiple: float = field(default=0.0)  # pnl / credit received (dollars)


def daily_realized_vol(bars: pd.DataFrame, lookback: int) -> pd.Series:
    """Annualized close-to-close vol per calendar date, using only days before it (no look-ahead)."""
    closes = bars["close"].groupby(bars.index.date).last()
    rv = np.log(closes).diff().rolling(lookback).std() * math.sqrt(252)
    return rv.shift(1)


def run_put30(bars: pd.DataFrame, cfg: Put30Config = Put30Config()) -> List[OptionTrade]:
    """`bars`: tz-aware (America/New_York), bar-open timestamps, OHLC of the front WTI future."""
    rv = daily_realized_vol(bars, cfg.rv_lookback_days)
    expiries = expiries_between(bars.index[0].date(), bars.index[-1].date())
    bar_len = bars.index.to_series().diff().median()

    trades: List[OptionTrade] = []
    last_cycle: Optional[date] = None
    pos = None  # dict while short

    for ts, row in zip(bars.index, bars.itertuples(index=False)):
        today = ts.date()
        t = ts.time()
        vol_today = rv.get(today, float("nan"))

        if pos is None:
            entry_start = datetime.combine(today, cfg.entry_time)
            in_window = entry_start <= ts.replace(tzinfo=None) < entry_start + timedelta(minutes=5)
            if not in_window or not (vol_today > 0):
                continue
            e0 = front_expiry(today, expiries)
            if e0 == last_cycle or (e0 - today).days < cfg.min_dte_entry:
                continue
            last_cycle = e0
            iv = vol_today * cfg.iv_mult
            f = row.open  # price at the 10:00 candle time
            tte = _years_to(ts, e0)
            k = black76.strike_for_delta(f, tte, iv, "P", cfg.target_delta, cfg.strike_tick)
            credit = black76.price(f, k, tte, iv, "P") - cfg.slippage
            pos = dict(entry_time=ts, expiry=e0, strike=k, f0=f, iv0=iv, credit=credit)
            continue

        # Mark at bar close
        now = ts + bar_len
        iv = pos["iv0"] if cfg.sticky_iv or not (vol_today > 0) else vol_today * cfg.iv_mult
        tte = _years_to(now, pos["expiry"])
        mark = black76.price(row.close, pos["strike"], tte, iv, "P")
        credit = pos["credit"]
        reason = None

        if now.time() >= cfg.entry_time or now.date() > pos["entry_time"].date():
            if mark <= cfg.take_profit_frac * credit:
                reason = "take_profit"
            elif cfg.stop_mult is not None and mark >= cfg.stop_mult * credit:
                reason = "stop"
        dte = (pos["expiry"] - now.date()).days
        if reason is None and dte <= 1 and now.time() >= cfg.exit_time:
            reason = "pre_expiry"
        if reason is None and tte <= 0:
            reason = "expired"  # data gap skipped the pre-expiry exit
        if reason is None and ts == bars.index[-1]:
            reason = "end_of_data"  # still open when the data runs out: mark it, don't drop it

        if reason:
            exit_px = mark + cfg.slippage if reason != "expired" else mark
            pnl_bbl = credit - exit_px
            pnl = pnl_bbl * cfg.multiplier * cfg.lots - 2 * cfg.commission * cfg.lots
            trades.append(OptionTrade(
                entry_time=pos["entry_time"], exit_time=now, expiry=pos["expiry"],
                strike=pos["strike"], entry_fut=pos["f0"], exit_fut=row.close,
                entry_iv=round(pos["iv0"], 4), credit=round(credit, 4), exit_price=round(exit_px, 4),
                reason=reason, pnl_dollars=round(pnl, 2),
                r_multiple=round(pnl / (credit * cfg.multiplier * cfg.lots), 3) if credit > 0 else 0.0,
            ))
            pos = None

    return trades


def _years_to(ts: pd.Timestamp, expiry: date) -> float:
    exp_dt = pd.Timestamp(datetime.combine(expiry, EXPIRY_TIME), tz=ts.tz)
    return max((exp_dt - ts).total_seconds(), 0.0) / (365.0 * 86400)
