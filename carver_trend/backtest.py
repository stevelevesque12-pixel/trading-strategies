"""Daily portfolio backtest for the Carver fast trend-following strategy."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .strategy import (
    Params,
    blended_sigma,
    buffer_width,
    buffered_target,
    capped_forecast,
    instrument_diversification_multiplier,
    optimal_position,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"


@dataclass(frozen=True)
class Spec:
    symbol: str
    file_glob: str
    multiplier: float
    tick_size: float
    commission: float  # per contract per side, USD
    rolls_per_year: int
    asset_class: str

    @property
    def cost_per_side(self) -> float:
        """Commission + 1 tick of slippage per contract per side."""
        return self.commission + self.tick_size * self.multiplier


SPECS: Dict[str, Spec] = {s.symbol: s for s in [
    Spec("ES", "real_es_15m_2016*", 50, 0.25, 2.25, 4, "Equity index"),
    Spec("NQ", "real_nq_15m_2016*", 20, 0.25, 2.25, 4, "Equity index"),
    Spec("GC", "real_gc_15m_2016*", 100, 0.10, 2.25, 6, "Metals"),
    Spec("SI", "real_si_15m_2016*", 5000, 0.005, 2.25, 5, "Metals"),
    Spec("MES", "real_mes_15m_*", 5, 0.25, 0.62, 4, "Equity index"),
    Spec("MNQ", "real_mnq_15m_*", 2, 0.25, 0.62, 4, "Equity index"),
    Spec("MYM", "real_mym_15m_*", 0.5, 1.0, 0.62, 4, "Equity index"),
    Spec("M2K", "real_m2k_15m_*", 5, 0.10, 0.62, 4, "Equity index"),
    Spec("MGC", "real_mgc_15m_*", 10, 0.10, 0.62, 6, "Metals"),
    Spec("SIL", "real_sil_15m_*", 1000, 0.005, 0.62, 5, "Metals"),
    Spec("MCL", "real_mcl_15m_*", 100, 0.01, 0.62, 12, "Energy"),
]}


def load_daily(spec: Spec) -> pd.DataFrame:
    """Resample intraday bars to daily sessions (18:00 ET -> 17:00 ET)."""
    path = next(DATA_DIR.glob(spec.file_glob + ".parquet"))
    bars = pd.read_parquet(path).tz_convert("America/New_York")
    session = (bars.index + pd.Timedelta(hours=6)).tz_localize(None).normalize()
    daily = bars.groupby(session).agg(open=("open", "first"), close=("close", "last"))
    return daily[daily.index.dayofweek < 5]


@dataclass
class Result:
    equity: pd.Series
    positions: pd.DataFrame
    forecasts: pd.DataFrame
    pnl_by_instrument: pd.DataFrame
    costs: pd.Series
    contracts_traded: pd.DataFrame
    symbols: List[str] = field(default_factory=list)


def run(symbols: List[str], params: Params = Params(), capital: float = 1_000_000,
        warmup_days: int = None, start: str = None, end: str = None) -> Result:
    specs = [SPECS[s] for s in symbols]
    daily = {s.symbol: load_daily(s) for s in specs}
    calendar = sorted(set().union(*(d.index for d in daily.values())))
    calendar = pd.DatetimeIndex(calendar)
    if end:
        calendar = calendar[calendar <= pd.Timestamp(end)]

    traded = pd.DataFrame({s: daily[s].reindex(calendar)["close"].notna() for s in symbols})
    close = pd.DataFrame({s: daily[s]["close"].reindex(calendar).ffill() for s in symbols})
    opens = pd.DataFrame({s: daily[s]["open"].reindex(calendar) for s in symbols})

    sigma = pd.DataFrame({s: blended_sigma(daily[s]["close"], params).reindex(calendar).ffill() for s in symbols})
    forecast = pd.DataFrame({s: capped_forecast(daily[s]["close"], sigma[s].reindex(daily[s].index), params)
                             .reindex(calendar).ffill() for s in symbols})

    if warmup_days is None:  # same as QC: 365 * blend_years + slow span + 7 calendar days
        warmup_days = 365 * params.blend_years + params.slow_ema_span + 7
    first_trade_day = calendar[0] + pd.Timedelta(days=warmup_days)
    if start:
        first_trade_day = max(first_trade_day, pd.Timestamp(start))
    calendar = calendar[calendar >= first_trade_day]

    idm = instrument_diversification_multiplier(len(symbols))
    weight = 1.0 / len(symbols)
    mult = {s.symbol: s.multiplier for s in specs}
    cost_side = {s.symbol: s.cost_per_side for s in specs}
    roll_cost_per_day = {s.symbol: 2 * s.cost_per_side * s.rolls_per_year / 256 for s in specs}

    equity = capital
    pos = {s: 0 for s in symbols}
    pending = {s: 0 for s in symbols}  # target set at yesterday's close
    rows_eq, rows_pos, rows_pnl, rows_cost, rows_traded = [], [], [], [], []
    prev_day = None

    for day in calendar:
        day_pnl = {}
        day_cost = 0.0
        day_traded = {}
        for s in symbols:
            if prev_day is None:
                day_pnl[s] = 0.0
                day_traded[s] = 0
                continue
            prev_close, c = close.at[prev_day, s], close.at[day, s]
            if traded.at[day, s] and pending[s] != pos[s]:
                o = opens.at[day, s]
                pnl = pos[s] * (o - prev_close) + pending[s] * (c - o)
                n = abs(pending[s] - pos[s])
                day_cost += n * cost_side[s]
                day_traded[s] = n
                pos[s] = pending[s]
            else:
                pnl = pos[s] * (c - prev_close)
                day_traded[s] = 0
            day_pnl[s] = pnl * mult[s]
            day_cost += abs(pos[s]) * roll_cost_per_day[s]

        equity += sum(day_pnl.values()) - day_cost

        for s in symbols:
            f, sg, px = forecast.at[day, s], sigma.at[day, s], close.at[day, s]
            if not np.isfinite(f) or not np.isfinite(sg) or sg <= 0:
                pending[s] = pos[s]
                continue
            n_opt = optimal_position(f, equity, idm, weight, mult[s], px, sg, params)
            b = buffer_width(equity, idm, weight, mult[s], px, sg, params)
            pending[s] = buffered_target(pos[s], n_opt, b)

        rows_eq.append(equity)
        rows_pos.append(dict(pos))
        rows_pnl.append(day_pnl)
        rows_cost.append(day_cost)
        rows_traded.append(day_traded)
        prev_day = day

    return Result(
        equity=pd.Series(rows_eq, index=calendar, name="equity"),
        positions=pd.DataFrame(rows_pos, index=calendar),
        forecasts=forecast.loc[calendar],
        pnl_by_instrument=pd.DataFrame(rows_pnl, index=calendar),
        costs=pd.Series(rows_cost, index=calendar),
        contracts_traded=pd.DataFrame(rows_traded, index=calendar),
        symbols=symbols,
    )


def summarize(res: Result, capital: float = 1_000_000) -> dict:
    eq = res.equity
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    f = res.forecasts
    return {
        "start": eq.index[0].date(), "end": eq.index[-1].date(), "years": round(years, 2),
        "final_equity": round(eq.iloc[-1]),
        "total_return_pct": round((eq.iloc[-1] / capital - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 2), "ann_vol_pct": round(vol * 100, 1),
        "sharpe": round(sharpe, 2), "max_dd_pct": round(dd * 100, 1),
        "calmar": round(cagr / abs(dd), 2) if dd < 0 else None,
        "costs_usd": round(res.costs.sum()),
        "contracts_per_year": round(res.contracts_traded.sum().sum() / years),
        "pct_days_long": round((f > 0).mean().mean() * 100, 1),
        "avg_abs_forecast": round(f.abs().mean().mean(), 1),
    }
