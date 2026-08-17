"""
VWAP standard-deviation band fade, gated by a rolling Hurst exponent
regime filter.

Runs on 1-minute bars (the Hurst window is specified in literal minutes,
so this strategy assumes 1-minute input -- see the module note below if
you resample).

1. **Session VWAP + stdev bands** — reuses `overextension.strategy.SessionVWAP`:
   a cumulative volume-weighted VWAP and a volume-weighted standard
   deviation around it, both resetting at the start of each session day
   ("strictly from the current session's volume distribution", per spec).
   Bands are `vwap +/- k * stdev` for any `k` -- this strategy only needs
   the `entry_sd` and `stop_sd` multiples, not literal 1st/2nd/3rd bands as
   separate objects.
2. **Rolling Hurst exponent** — a classic single-window R/S (rescaled
   range) estimate, `H = log(R/S) / log(N)`, over the trailing
   `hurst_window` bars (default 30 -- 30 minutes on a 1-minute chart).
   This is a lightweight, single-scale estimator (not a more statistically
   robust multi-scale regression) -- appropriate for a fast rolling filter,
   same "simple over academically rigorous" tradeoff this repo already
   makes for ORB's ATR (simple MA of True Range, not Wilder's smoothing).
   Unlike the VWAP bands, the Hurst window deliberately does **not** reset
   at session start -- it carries over from the prior session's trailing
   bars, same reasoning as ORB's trailing ATR (a 30-bar window reset at
   9:30 wouldn't be usable again until 9:30 + 30 minutes every single day).
   H < 0.5 => mean-reverting/range-bound; H > 0.5 => trending; H = 0.5 =>
   a random walk. Known caveat: single-window R/S estimates have a
   documented small-sample bias (they skew low at N~30), so treat
   `hurst_threshold` as a tunable knob to backtest, not a literal
   textbook 0.5 -- it consistently reads a true random walk as somewhat
   mean-reverting at this window size.
3. **Entry** (only while `H < hurst_threshold`, i.e. the regime is
   confirmed range-bound): the spec describes fading the upper band short
   -- this port mirrors that to the lower band for symmetry (long), same
   as every other mean-reversion strategy in this repo trades both sides.
   Short when `bar.high >= vwap + entry_sd * stdev` (spec default
   entry_sd=2.5); long when `bar.low <= vwap - entry_sd * stdev`. Like
   ORB's breakout triggers, this fires at the exact touched band level on
   the bar that reaches it, not at that bar's close. If a single bar
   improbably touches both bands, short is checked first (documented
   tie-break, arbitrary but consistent).
4. **Hard stop** — just outside the `stop_sd` band (default 3.0, i.e. the
   3rd standard deviation), plus a small tick buffer: "an automatic
   stop-loss ... to prevent catastrophic losses if the market suddenly
   shifts into an aggressive trend." `stop_sd` must be greater than
   `entry_sd` (checked at construction) -- a stop inside the entry trigger
   is a contradiction, not a valid config.
5. **Target** — the VWAP baseline itself, frozen at entry time (not the
   live-updating VWAP): "targeting a reversion back to VWAP baseline."
   Freezing it is what lets this reuse the same static-stop/target
   backtest engine as overextension/ORB rather than needing a strategy
   that manages its own trade lifecycle bar-by-bar.
6. **Multiple trades per day allowed** — unlike ORB's one-shot breakout,
   a range-bound session can bounce off the bands more than once; there's
   no per-day trade cap here, same as overextension.
7. **Intraday only** — same session-reset/entry-window/flatten-cutoff
   convention as the rest of this repo (`SessionConfig`, shared): no new
   entries at/after `no_entry_after` (default 15:45 ET), any open position
   force-flattened at `flatten_at` (default 15:55 ET).
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from overextension.strategy import SessionVWAP


class RollingHurst:
    """Rolling single-window R/S (rescaled range) Hurst exponent estimate,
    recomputed from scratch over the trailing `window` log returns each
    update (O(window) per bar -- negligible at window~30). Does NOT reset
    per session; see the module docstring for why."""

    def __init__(self, window: int = 30):
        self.window = window
        self._closes: deque = deque(maxlen=window + 1)

    def update(self, close: float) -> None:
        self._closes.append(close)

    @property
    def value(self) -> Optional[float]:
        if len(self._closes) < self.window + 1:
            return None
        closes = list(self._closes)
        returns = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)]
        n = len(returns)
        mean_r = sum(returns) / n
        deviations = [r - mean_r for r in returns]

        cumulative = []
        running = 0.0
        for d in deviations:
            running += d
            cumulative.append(running)
        r = max(cumulative) - min(cumulative)

        variance = sum(d * d for d in deviations) / n
        s = variance**0.5

        if s == 0 or r == 0:
            return None  # degenerate window (flat prices) -- can't estimate
        return math.log(r / s) / math.log(n)


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


class HurstVWAPStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        entry_sd: float = 2.5,
        stop_sd: float = 3.0,
        stop_buffer_ticks: int = 2,
        hurst_window: int = 30,
        hurst_threshold: float = 0.5,
        min_stdev_ticks: float = 2.0,
        warmup_bars: int = 20,
        session: Optional[SessionConfig] = None,
    ):
        if stop_sd <= entry_sd:
            raise ValueError(
                f"stop_sd={stop_sd} must be greater than entry_sd={entry_sd} -- "
                "the hard stop has to sit outside the entry band, not inside it."
            )

        self.tick_size = tick_size
        self.entry_sd = entry_sd
        self.stop_sd = stop_sd
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.hurst_threshold = hurst_threshold
        self.min_stdev_ticks = min_stdev_ticks
        self.warmup_bars = warmup_bars
        self.session = session or SessionConfig()

        self.vwap = SessionVWAP()
        self.hurst = RollingHurst(hurst_window)
        self._bar_count = 0
        self._current_date = None

    def reset(self) -> None:
        """Clear the session VWAP/stdev state -- called automatically on a
        new session day. Deliberately does NOT reset the rolling Hurst window."""
        self.vwap.reset()
        self._bar_count = 0

    def _roll_session(self, ts) -> None:
        d = ts.date()
        if self._current_date is None:
            self._current_date = d
        elif d != self._current_date:
            self._current_date = d
            self.reset()

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.session_start <= t < self.session.no_entry_after

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        self.vwap.update(bar)
        self.hurst.update(bar.close)
        self._bar_count += 1

        vwap = self.vwap.vwap
        sd = self.vwap.stdev
        if vwap is None or sd is None or self._bar_count < self.warmup_bars:
            return None
        sd = max(sd, self.min_stdev_ticks * self.tick_size)

        if not self._in_entry_window(bar.timestamp):
            return None

        h = self.hurst.value
        if h is None or h >= self.hurst_threshold:
            return None  # regime not confirmed range-bound (or not enough history yet)

        short_trigger = vwap + self.entry_sd * sd
        long_trigger = vwap - self.entry_sd * sd

        if bar.high >= short_trigger:
            return self._fire("short", bar, short_trigger, vwap, sd, h)
        if bar.low <= long_trigger:
            return self._fire("long", bar, long_trigger, vwap, sd, h)
        return None

    def _fire(self, direction: str, bar: Bar, entry_price: float, vwap: float, sd: float, h: float) -> Optional[Signal]:
        if direction == "short":
            stop_price = vwap + self.stop_sd * sd + self.stop_buffer
            risk = stop_price - entry_price
        else:
            stop_price = vwap - self.stop_sd * sd - self.stop_buffer
            risk = entry_price - stop_price

        if risk <= 0:
            # Only reachable with a degenerate config (min_stdev_ticks=0 and
            # stop_buffer_ticks=0) on a perfectly flat window (stdev==0).
            return None

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=vwap,  # frozen at entry -- "reversion back to VWAP baseline"
            reason=f"H={h:.2f} vwap={vwap:.2f} sd={sd:.2f} trigger={entry_price:.2f}",
        )
