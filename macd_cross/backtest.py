"""
Bar-by-bar backtest of the MACD crossover strategy.

Execution model
- Signal on bar t's close, market fill at bar t+1's open (1 bar latency).
- Stop-and-reverse: a buy signal closes any short and goes long; a sell
  signal closes any long and goes short. Always 1 position max.
- Costs per side: `commission` dollars + `slippage_ticks` ticks.

Session modes
- "24h": trade the full Globex session; positions ride through the daily
  maintenance break and weekends (the classic always-in-the-market reading
  of the rule).
- "rth": intraday only. Entries only on signals from bars that close in
  [09:30, 15:45) ET; any open position is flattened at the open of the
  first bar at/after 15:55 ET (or the last bar of the day if none).
"""

from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import pandas as pd

from failed2s.instruments import Instrument

from .strategy import MACDParams, crossover_signals


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_points: float
    gross_dollars: float
    cost_dollars: float
    net_dollars: float


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.62  # $ per contract per side (typical MES all-in)
    slippage_ticks: float = 1.0  # ticks per side

    def round_turn(self, inst: Instrument, contracts: int = 1) -> float:
        per_side = self.commission + self.slippage_ticks * inst.tick_size * inst.point_value
        return 2 * per_side * contracts


@dataclass(frozen=True)
class SessionRules:
    mode: str = "24h"  # "24h" or "rth"
    entry_start: time = time(9, 30)
    entry_end: time = time(15, 45)
    flatten_at: time = time(15, 55)


def run_backtest(
    bars: pd.DataFrame,
    bar_len: pd.Timedelta,
    inst: Instrument,
    params: MACDParams = MACDParams(),
    costs: CostModel = CostModel(),
    rules: SessionRules = SessionRules(),
    contracts: int = 1,
) -> List[Trade]:
    """`bars` must be indexed by bar *open* time in America/New_York."""
    signals = crossover_signals(bars["close"], params).to_numpy()
    opens = bars["open"].to_numpy()
    closes = bars["close"].to_numpy()
    idx = bars.index
    close_times = idx + bar_len
    rt_cost = costs.round_turn(inst, contracts)
    rth = rules.mode == "rth"

    trades: List[Trade] = []
    pos = 0
    entry_px = 0.0
    entry_t: Optional[pd.Timestamp] = None

    def close_pos(px: float, t: pd.Timestamp, reason: str) -> None:
        nonlocal pos
        pts = (px - entry_px) * pos
        gross = pts * inst.point_value * contracts
        trades.append(Trade(entry_t, t, pos, entry_px, px, reason, pts, gross, rt_cost, gross - rt_cost))
        pos = 0

    n = len(bars)
    for i in range(n):
        t = idx[i]
        # 1) RTH flatten: at the open of the first bar at/after flatten_at,
        #    or when the calendar day rolls without having hit it.
        if rth and pos != 0 and (t.date() != entry_t.date() or t.time() >= rules.flatten_at):
            if t.date() != entry_t.date():
                close_pos(closes[i - 1], idx[i - 1] + bar_len, "session_flatten")
            else:
                close_pos(opens[i], t, "session_flatten")

        # 2) Act on the previous bar's signal at this bar's open.
        if i == 0:
            continue
        s = signals[i - 1]
        if s == 0 or s == pos:
            continue
        if rth:
            ct = close_times[i - 1].time()
            if not (rules.entry_start < ct <= rules.entry_end) or t.time() >= rules.flatten_at \
                    or t.date() != idx[i - 1].date():
                # Outside the entry window: still honour an exit signal.
                if pos != 0 and s == -pos:
                    close_pos(opens[i], t, "signal_exit")
                continue
        if pos != 0:
            close_pos(opens[i], t, "reverse")
        pos, entry_px, entry_t = s, opens[i], t

    if pos != 0:
        close_pos(closes[-1], idx[-1] + bar_len, "end_of_data")
    return trades


def metrics(trades: List[Trade], days: int) -> dict:
    if not trades:
        return {"trades": 0}
    net = pd.Series([t.net_dollars for t in trades])
    gross = pd.Series([t.gross_dollars for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    equity = net.cumsum()
    dd = (equity.cummax().clip(lower=0) - equity).max()
    return {
        "trades": len(trades),
        "trades_per_day": round(len(trades) / max(days, 1), 1),
        "win_rate_pct": round(len(wins) / len(net) * 100, 1),
        "gross_pnl": round(gross.sum(), 2),
        "costs": round(sum(t.cost_dollars for t in trades), 2),
        "net_pnl": round(net.sum(), 2),
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else float("inf"),
        "avg_trade": round(net.mean(), 2),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "max_drawdown": round(dd, 2),
    }
