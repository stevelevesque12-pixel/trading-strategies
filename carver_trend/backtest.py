"""
Offline re-implementation of the QuantConnect AdjustedTrendFuturesAlgorithm
(quantconnect_main.py) on Rob Carver's back-adjusted daily futures data, so it
can be run across all major US futures without LEAN.

Follows "Advanced Futures Trading Strategies" (Carver, 2023):
  * EWMAC(f, 4f) trend filters for f = 2, 4, ..., 2**emac_filters, each divided by
    daily price-point vol, scaled with Table 29's forecast scalars, capped at +-20
  * forecasts averaged, x FDM (Table 52), capped at +-20
  * "adjusted": Strategy 13 volatility-regime multiplier. Relative vol V = sigma% /
    its 10-year mean, Q = V's percentile in its own history, M = EWMA_10(2 - 1.5Q),
    forecast x M then re-capped (disable with --no-regime for plain Strategy 9)
  * sigma = 0.7 * EWMA(span=sigma_span) + 0.3 * mean of it over blend_years
  * position = forecast/10 * capital * IDM * weight * target_risk / (multiplier * sigma_annual)
  * buffered integer positions: only trade when outside +-buffer_scaler * average position
  * costs: half-spread + commission per contract traded, plus rolls

Decision on day t's close, executed at day t+1's close (one-day lag).
Capital is held fixed (no compounding), as in the book.

    python carver_trend/backtest.py                       # full history 2000-2024
    python carver_trend/backtest.py --start 2019-03-28    # QC's 5-year window
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "sample_data" / "carver_daily"
DAYS = 256
FORECAST_SCALAR = {2: 12.1, 4: 8.53, 8: 5.95, 16: 4.10, 32: 2.79, 64: 1.91}  # Table 29, p.177
FDM = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}                     # Table 52, p.234


def idm(n: int) -> float:
    """Instrument diversification multiplier, Table 16 (p.76)."""
    for limit, value in [(1, 1.0), (2, 1.2), (3, 1.48), (4, 1.56), (5, 1.7), (6, 1.9), (7, 2.1),
                         (14, 2.2), (24, 2.3), (29, 2.4)]:
        if n <= limit:
            return value
    return 2.5


def forecast(adj: pd.Series, price: pd.Series, p) -> tuple[pd.Series, pd.Series]:
    """Return (combined capped forecast, blended daily sigma in price points)."""
    diff = adj.diff()
    sigma = diff.ewm(span=p.sigma_span, min_periods=p.sigma_span).std()
    sigma = 0.7 * sigma + 0.3 * sigma.rolling(DAYS * p.blend_years, min_periods=DAYS).mean()
    sigma = sigma.clip(lower=1e-9)

    spans = [2 ** i for i in range(1, p.emac_filters + 1)]
    parts = []
    for f in spans:
        raw = (adj.ewm(span=f, adjust=False).mean() - adj.ewm(span=4 * f, adjust=False).mean()) / sigma
        parts.append((raw * FORECAST_SCALAR[f]).clip(-p.cap, p.cap))
    fc = (pd.concat(parts, axis=1).mean(axis=1) * FDM[len(spans)]).clip(-p.cap, p.cap)

    if p.regime:
        pct_vol = (sigma / price.abs()).replace([np.inf, -np.inf], np.nan)
        rel = pct_vol / pct_vol.rolling(DAYS * 10, min_periods=DAYS).mean()
        q = rel.expanding(min_periods=DAYS).rank(pct=True)
        mult = (2 - 1.5 * q).ewm(span=10).mean()
        fc = (fc * mult).clip(-p.cap, p.cap)
    return fc, sigma


def run(p) -> dict:
    prices = pd.read_parquet(DATA / "adjusted_daily.parquet")
    meta = pd.read_csv(DATA / "instruments.csv").set_index("instrument")
    if p.instruments:
        meta = meta.loc[p.instruments]
    adj = prices.pivot(index="date", columns="instrument", values="adj")[meta.index]
    cur = prices.pivot(index="date", columns="instrument", values="price")[meta.index]
    adj = adj[adj.index.dayofweek < 5]
    cur = cur.reindex(adj.index)

    fcs, sigmas = {}, {}
    for inst in meta.index:
        a = adj[inst].dropna()
        f, s = forecast(a, cur[inst].reindex(a.index).ffill(), p)
        fcs[inst], sigmas[inst] = f, s
    fc = pd.DataFrame(fcs).reindex(adj.index).ffill()
    sigma = pd.DataFrame(sigmas).reindex(adj.index).ffill()
    # Quiet the ffill where an instrument hasn't started yet / has stopped trading.
    live = adj.notna().cumsum().gt(0) & adj.bfill().notna()
    # QC warm-up: max(365*blend_years, 4*2**emac_filters) days of history before trading.
    warm = max(DAYS * p.blend_years, 4 * 2 ** p.emac_filters)
    ready = live & (adj.notna().cumsum() > warm) & fc.notna() & sigma.notna()

    idx = adj.index[(adj.index >= p.start) & (adj.index <= p.end)]
    n_live = ready.loc[idx].sum(axis=1)
    mult = meta["multiplier"]
    risk_per_inst = p.capital * n_live.map(idm) / n_live.replace(0, np.nan) * p.target_risk
    avg_pos = (1 / (mult * sigma.loc[idx] * 16)).mul(risk_per_inst, axis=0).where(ready.loc[idx])
    optimal = (avg_pos * fc.loc[idx] / 10).fillna(0.0)
    buffer = (avg_pos.abs() * p.buffer_scaler).fillna(0.0)

    # Buffered positions decided at each close (path dependent -> loop over days).
    opt, buf = optimal.to_numpy(), buffer.to_numpy()
    decided = np.zeros_like(opt)
    pos = np.zeros(opt.shape[1])
    for t in range(len(idx)):
        lo, hi = opt[t] - buf[t], opt[t] + buf[t]
        if not p.fractional:
            lo, hi = np.round(lo), np.round(hi)
        pos = np.where(pos < lo, lo, np.where(pos > hi, hi, pos))
        decided[t] = pos
    decided = pd.DataFrame(decided, index=idx, columns=meta.index)
    held = decided.shift(1).fillna(0.0)           # traded at next day's close
    traded = held.diff().abs().fillna(held.abs())

    dpx = adj.loc[idx].diff().fillna(0.0)          # P&L over (t-1, t] from position held into t
    gross = held.shift(1).fillna(0.0) * dpx * mult
    per_contract = meta["spread"] * mult + meta["commission"]
    cost = traded * per_contract + held.abs() * per_contract * 2 * meta["rolls_per_year"] / DAYS
    net = gross - cost
    return dict(net=net, gross=gross, cost=cost, held=held, traded=traded, fc=fc.loc[idx],
                n_live=n_live, meta=meta, optimal=optimal)


def stats(pnl: pd.Series, capital: float) -> dict:
    r = pnl / capital
    r = r[r.index >= r.ne(0).idxmax()]
    eq = r.cumsum()
    dd = eq - eq.cummax()
    weekly = r.resample("W").sum()
    return {
        "ann_return_%": 100 * r.mean() * DAYS,
        "ann_vol_%": 100 * r.std() * 16,
        "sharpe": r.mean() / r.std() * 16 if r.std() > 0 else np.nan,
        "max_dd_%": 100 * dd.min(),
        "skew_weekly": weekly.skew(),
        "worst_day_%": 100 * r.min(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--emac-filters", dest="emac_filters", type=int, default=6)
    ap.add_argument("--cap", type=float, default=20)
    ap.add_argument("--sigma-span", dest="sigma_span", type=int, default=32)
    ap.add_argument("--target-risk", dest="target_risk", type=float, default=0.2)
    ap.add_argument("--blend-years", dest="blend_years", type=int, default=3)
    ap.add_argument("--buffer-scaler", dest="buffer_scaler", type=float, default=0.1)
    ap.add_argument("--no-regime", dest="regime", action="store_false")
    ap.add_argument("--fractional", action="store_true", help="allow fractional contracts")
    ap.add_argument("--instruments", nargs="*")
    p = ap.parse_args()

    res = run(p)
    net, meta = res["net"], res["meta"]
    s = stats(net.sum(axis=1), p.capital)
    print(f"{net.index[0].date()} -> {net.index[-1].date()}, instruments live at end: {int(res['n_live'].iloc[-1])}")
    print("portfolio:", {k: round(v, 2) for k, v in s.items()})
    print(f"costs: {100 * res['cost'].sum().sum() / p.capital / (len(net) / DAYS):.2f}% of capital per year")

    by_inst = pd.DataFrame({
        "class": meta["asset_class"],
        "pnl_$k": net.sum() / 1e3,
        "sharpe": net.mean() / net.std() * 16,
        "avg_|contracts|": res["held"].abs().mean(),
        "trades/yr": (res["traded"] > 0).sum() / (len(net) / DAYS),
    }).sort_values("pnl_$k", ascending=False)
    print(by_inst.round(2).to_string())
    by_class = net.T.groupby(meta["asset_class"]).sum().T
    print("\nby asset class ($k):", (by_class.sum() / 1e3).round(0).to_dict())
    yearly = net.sum(axis=1).resample("YE").sum() / p.capital * 100
    print("\nyearly %:", {d.year: round(v, 1) for d, v in yearly.items()})


if __name__ == "__main__":
    main()
