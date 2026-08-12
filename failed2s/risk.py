"""Prop-firm-style risk management: daily loss limit and trade-count caps."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskManager:
    daily_loss_limit: float
    max_daily_trades: int = 3

    _daily_pnl: float = field(default=0.0, init=False, repr=False)
    _daily_trades: int = field(default=0, init=False, repr=False)
    _current_date: Optional[object] = field(default=None, init=False, repr=False)
    _locked: bool = field(default=False, init=False, repr=False)

    def _roll_day(self, date) -> None:
        if date != self._current_date:
            self._current_date = date
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._locked = False

    def can_enter(self, date) -> bool:
        self._roll_day(date)
        if self._locked:
            return False
        if self._daily_trades >= self.max_daily_trades:
            return False
        return True

    def record_trade(self, date, pnl: float) -> None:
        self._roll_day(date)
        self._daily_pnl += pnl
        self._daily_trades += 1
        if self._daily_pnl <= -abs(self.daily_loss_limit):
            self._locked = True

    @property
    def locked_out(self) -> bool:
        return self._locked
