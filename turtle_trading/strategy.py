"""
Turtle Trading -- Richard Dennis / William Eckhardt's 1983 trend-following
rules, as documented in Curtis Faith's "The Original Turtle Trading Rules".

Runs on DAILY bars. Per market:

  - N = 20-day Wilder ATR (first N is the simple mean of the first 20 true
    ranges, then N = (19 * prev_N + TR) / 20).
  - Unit size = floor(equity * risk_pct / (N * point_value)), so a 1N move
    against one unit costs `risk_pct` (default 1%) of equity.
  - System 1: enter on a 20-day breakout (price trades beyond the prior 20
    days' high/low). Skip it if the *last* 20-day breakout would have been a
    winner -- tracked hypothetically, whether or not it was actually taken.
    Failsafe: if skipped, still enter on a 55-day breakout so a major trend
    isn't missed. Exit on a 10-day opposite breakout.
  - System 2: enter on every 55-day breakout; exit on a 20-day opposite
    breakout.
  - Stop: 2N from entry. Pyramid another unit every +0.5N from the previous
    unit's fill, up to `max_units` (4); each add moves the stop for ALL units
    to 2N from the newest fill. N is frozen at the initial entry.

Daily-bar fill model (no intraday data, so ordering within a bar is unknown):
  - Stop-type orders fill at the trigger level, or at the open if the bar
    gapped through it, plus `slippage_ticks` against us.
  - Existing positions check exits before pyramiding. After an entry or add,
    the (new) stop and exit channel are re-checked on the same bar and
    assumed hit if the bar's range reaches them -- the conservative reading.
  - A bar that breaks BOTH the upper and lower channel is ambiguous and is
    ignored for entries.
  - No new entry on the same bar a position was closed.

Not modeled: the Turtles' portfolio-level unit caps (12 per direction, 6 per
correlated group), their 10%-drawdown equity haircut, and roll costs on
continuous-contract data.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from failed2s.bars import Bar
from failed2s.instruments import Instrument

Direction = Literal["long", "short"]


@dataclass
class TurtleConfig:
    system: int = 1  # 1 = 20-day entry / 10-day exit, 2 = 55-day entry / 20-day exit
    atr_len: int = 20
    risk_pct: float = 0.01
    stop_n: float = 2.0
    add_n: float = 0.5
    max_units: int = 4
    skip_after_winner: bool = True  # System 1 only
    failsafe_len: int = 55  # System 1 only
    slippage_ticks: float = 0.0
    commission_per_contract: float = 0.0  # round trip

    def __post_init__(self):
        if self.system not in (1, 2):
            raise ValueError("system must be 1 or 2")

    @property
    def entry_len(self) -> int:
        return 20 if self.system == 1 else 55

    @property
    def exit_len(self) -> int:
        return 10 if self.system == 1 else 20


class Account:
    """Realized-equity account; shared across markets in a portfolio run."""

    def __init__(self, starting_equity: float):
        self.starting_equity = starting_equity
        self.equity = starting_equity


@dataclass
class TurtleTrade:
    symbol: str
    system: int
    entry_reason: str
    direction: Direction
    entry_time: object
    exit_time: object
    n_at_entry: float
    units: int
    contracts: int
    avg_entry_price: float
    initial_stop: float
    exit_price: float
    exit_reason: str
    pnl_dollars: float
    r_multiple: float  # P&L in units of one unit's initial 2N risk
    equity_after: float


@dataclass
class _Position:
    direction: Direction
    entry_reason: str
    entry_time: object
    n: float
    unit_contracts: int
    fills: List[float] = field(default_factory=list)
    stop: float = 0.0
    initial_stop: float = 0.0

    @property
    def last_fill(self) -> float:
        return self.fills[-1]


@dataclass
class _Hypothetical:
    """A single-unit, unpyramided shadow of a System 1 breakout trade."""

    direction: Direction
    entry: float
    stop: float


def _sign(direction: Direction) -> int:
    return 1 if direction == "long" else -1


class TurtleSystem:
    """Runs one Turtle system on one market. Feed daily bars via on_bar()."""

    def __init__(self, instrument: Instrument, config: Optional[TurtleConfig] = None, account: Optional[Account] = None):
        self.instrument = instrument
        self.config = config or TurtleConfig()
        self.account = account or Account(100_000.0)

        lookback = max(self.config.entry_len, self.config.exit_len, self.config.failsafe_len)
        self._highs: deque = deque(maxlen=lookback)
        self._lows: deque = deque(maxlen=lookback)
        self._prev_close: Optional[float] = None
        self._tr_seed: List[float] = []
        self.n: Optional[float] = None

        self.position: Optional[_Position] = None
        self.trades: List[TurtleTrade] = []
        self.skipped_breakouts = 0  # S1 breakouts filtered because the last one won
        self.skipped_too_small = 0  # breakouts where 1 contract risked more than risk_pct

        self._hypo: Optional[_Hypothetical] = None
        self._last_hypo_won = False

    # ------------------------------------------------------------------ levels

    def _ready(self) -> bool:
        return self.n is not None and len(self._highs) >= self._highs.maxlen

    def _channel_high(self, length: int) -> float:
        return max(list(self._highs)[-length:])

    def _channel_low(self, length: int) -> float:
        return min(list(self._lows)[-length:])

    def _slip(self, price: float, direction: Direction, entering: bool) -> float:
        # Entering long / exiting short buys (pay up); the reverse sells.
        buying = (direction == "long") == entering
        s = self.config.slippage_ticks * self.instrument.tick_size
        return price + s if buying else price - s

    # --------------------------------------------------------------- main loop

    def on_bar(self, bar: Bar) -> None:
        if self._ready():
            self._process(bar)
        self._update_indicators(bar)

    def _update_indicators(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self._prev_close), abs(bar.low - self._prev_close))
        n_len = self.config.atr_len
        if self.n is None:
            self._tr_seed.append(tr)
            if len(self._tr_seed) == n_len:
                self.n = sum(self._tr_seed) / n_len
        else:
            self.n = ((n_len - 1) * self.n + tr) / n_len
        self._prev_close = bar.close
        self._highs.append(bar.high)
        self._lows.append(bar.low)

    def _process(self, bar: Bar) -> None:
        cfg = self.config
        exit_hi = self._channel_high(cfg.exit_len)
        exit_lo = self._channel_low(cfg.exit_len)

        closed_today = False
        if self.position is not None:
            closed_today = self._check_exit(bar, exit_hi, exit_lo, allow_gap=True)
            if not closed_today and self._pyramid(bar, allow_gap=True):
                closed_today = self._check_exit(bar, exit_hi, exit_lo, allow_gap=False)

        entry_hi = self._channel_high(cfg.entry_len)
        entry_lo = self._channel_low(cfg.entry_len)
        broke_up = bar.high > entry_hi
        broke_dn = bar.low < entry_lo
        breakout: Optional[Direction] = None
        if broke_up != broke_dn:
            breakout = "long" if broke_up else "short"

        take_primary = breakout is not None
        if cfg.system == 1 and cfg.skip_after_winner:
            take_primary = self._update_hypothetical(bar, breakout, entry_hi, entry_lo, exit_hi, exit_lo)

        if self.position is not None or closed_today:
            return

        if breakout is not None and take_primary:
            level = entry_hi if breakout == "long" else entry_lo
            self._enter(bar, breakout, level, f"breakout_{cfg.entry_len}", exit_hi, exit_lo)
        elif breakout is not None and cfg.system == 1:
            fs_hi = self._channel_high(cfg.failsafe_len)
            fs_lo = self._channel_low(cfg.failsafe_len)
            if breakout == "long" and bar.high > fs_hi:
                self._enter(bar, "long", fs_hi, f"failsafe_{cfg.failsafe_len}", exit_hi, exit_lo)
            elif breakout == "short" and bar.low < fs_lo:
                self._enter(bar, "short", fs_lo, f"failsafe_{cfg.failsafe_len}", exit_hi, exit_lo)

    # ------------------------------------------------- System 1 skip filter

    def _update_hypothetical(self, bar, breakout, entry_hi, entry_lo, exit_hi, exit_lo) -> bool:
        """
        Advance the shadow trade and return whether a breakout on this bar
        should be taken. The shadow is a single unit with a 2N stop and the
        System 1 exit channel, so "winner" means it reached a profitable
        10-day exit before being stopped out.
        """
        h = self._hypo
        if h is not None:
            if h.direction == "long":
                level = max(h.stop, exit_lo)
                if bar.low <= level:
                    self._last_hypo_won = min(bar.open, level) > h.entry
                    self._hypo = None
            else:
                level = min(h.stop, exit_hi)
                if bar.high >= level:
                    self._last_hypo_won = max(bar.open, level) < h.entry
                    self._hypo = None
            if self._hypo is not None:
                return False  # still inside the previous breakout's run

        if breakout is None:
            return False
        if self._last_hypo_won:
            self.skipped_breakouts += 1
        level = entry_hi if breakout == "long" else entry_lo
        entry = max(bar.open, level) if breakout == "long" else min(bar.open, level)
        stop = entry - _sign(breakout) * self.config.stop_n * self.n
        self._hypo = _Hypothetical(breakout, entry, stop)
        return not self._last_hypo_won

    # -------------------------------------------------------- order handling

    def _enter(self, bar: Bar, direction: Direction, level: float, reason: str, exit_hi: float, exit_lo: float) -> None:
        cfg = self.config
        dollar_vol = self.n * self.instrument.point_value
        unit = math.floor(self.account.equity * cfg.risk_pct / dollar_vol) if dollar_vol > 0 else 0
        if unit < 1:
            self.skipped_too_small += 1
            return

        raw = max(bar.open, level) if direction == "long" else min(bar.open, level)
        fill = self._slip(raw, direction, entering=True)
        stop = fill - _sign(direction) * cfg.stop_n * self.n
        self.position = _Position(direction, reason, bar.timestamp, self.n, unit, [fill], stop, stop)
        self._pyramid(bar, allow_gap=False)
        self._check_exit(bar, exit_hi, exit_lo, allow_gap=False)

    def _pyramid(self, bar: Bar, allow_gap: bool) -> bool:
        pos = self.position
        cfg = self.config
        added = False
        while len(pos.fills) < cfg.max_units:
            step = cfg.add_n * pos.n
            if pos.direction == "long":
                target = pos.last_fill + step
                if bar.high < target:
                    break
                raw = max(bar.open, target) if allow_gap else target
            else:
                target = pos.last_fill - step
                if bar.low > target:
                    break
                raw = min(bar.open, target) if allow_gap else target
            pos.fills.append(self._slip(raw, pos.direction, entering=True))
            pos.stop = pos.last_fill - _sign(pos.direction) * cfg.stop_n * pos.n
            added = True
        return added

    def _check_exit(self, bar: Bar, exit_hi: float, exit_lo: float, allow_gap: bool) -> bool:
        pos = self.position
        if pos.direction == "long":
            level = max(pos.stop, exit_lo)
            if bar.low > level:
                return False
            raw = min(bar.open, level) if allow_gap else level
        else:
            level = min(pos.stop, exit_hi)
            if bar.high < level:
                return False
            raw = max(bar.open, level) if allow_gap else level
        reason = "stop" if level == pos.stop else f"exit_{self.config.exit_len}"
        self._close(self._slip(raw, pos.direction, entering=False), bar.timestamp, reason)
        return True

    def close_open_position(self, price: float, timestamp, reason: str = "end_of_data") -> None:
        if self.position is not None:
            self._close(price, timestamp, reason)

    def _close(self, exit_price: float, timestamp, reason: str) -> None:
        pos = self.position
        pv = self.instrument.point_value
        sign = _sign(pos.direction)
        contracts = pos.unit_contracts * len(pos.fills)
        gross = sum((exit_price - f) * sign for f in pos.fills) * pos.unit_contracts * pv
        pnl = gross - contracts * self.config.commission_per_contract
        unit_risk = self.config.stop_n * pos.n * pv * pos.unit_contracts
        self.account.equity += pnl

        self.trades.append(
            TurtleTrade(
                symbol=self.instrument.symbol,
                system=self.config.system,
                entry_reason=pos.entry_reason,
                direction=pos.direction,
                entry_time=pos.entry_time,
                exit_time=timestamp,
                n_at_entry=pos.n,
                units=len(pos.fills),
                contracts=contracts,
                avg_entry_price=sum(pos.fills) / len(pos.fills),
                initial_stop=pos.initial_stop,
                exit_price=exit_price,
                exit_reason=reason,
                pnl_dollars=pnl,
                r_multiple=pnl / unit_risk if unit_risk else 0.0,
                equity_after=self.account.equity,
            )
        )
        self.position = None
