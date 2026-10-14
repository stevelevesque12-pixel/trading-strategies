"""
Download Rob Carver's back-adjusted daily futures prices (pysystemtrade repo) for
the instruments in UNIVERSE, collapse them to one row per day, and save a single
parquet for backtest.py.

The source CSVs are ~190MB in total (they contain some intraday rows near the end),
so only the daily parquet is kept in the repo. Coverage ends 2024-03-28.

    python carver_trend/fetch_data.py
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/robcarver17/pysystemtrade/master/data/futures"
OUT = Path(__file__).resolve().parent.parent / "sample_data" / "carver_daily"

# pysystemtrade code -> (asset class, contract description). USD-denominated, US-listed.
UNIVERSE = {
    "SP500": ("Equity", "E-mini S&P 500"),
    "NASDAQ": ("Equity", "E-mini Nasdaq-100"),
    "DOW": ("Equity", "Micro E-mini Dow"),
    "RUSSELL": ("Equity", "Micro E-mini Russell 2000"),
    "VIX": ("Vol", "VIX"),
    "US2": ("Bond", "2y T-Note"),
    "US5": ("Bond", "5y T-Note"),
    "US10": ("Bond", "10y T-Note"),
    "US20": ("Bond", "T-Bond"),
    "US30": ("Bond", "Ultra T-Bond"),
    "EUR": ("FX", "Euro FX"),
    "JPY": ("FX", "Japanese Yen"),
    "GBP": ("FX", "British Pound"),
    "AUD": ("FX", "Australian Dollar"),
    "CAD": ("FX", "Canadian Dollar"),
    "CHF": ("FX", "Swiss Franc"),
    "MXP": ("FX", "Mexican Peso"),
    "CRUDE_W": ("Energy", "WTI Crude"),
    "HEATOIL": ("Energy", "Heating Oil"),
    "GASOILINE": ("Energy", "RBOB Gasoline"),
    "GAS_US": ("Energy", "Natural Gas"),
    "GOLD": ("Metals", "Gold"),
    "SILVER": ("Metals", "Silver"),
    "COPPER": ("Metals", "Copper"),
    "PLAT": ("Metals", "Platinum"),
    "PALLAD": ("Metals", "Palladium"),
    "CORN": ("Ags", "Corn"),
    "WHEAT": ("Ags", "Wheat"),
    "SOYBEAN": ("Ags", "Soybeans"),
    "SOYMEAL": ("Ags", "Soybean Meal"),
    "SOYOIL": ("Ags", "Soybean Oil"),
    "LIVECOW": ("Ags", "Live Cattle"),
    "FEEDCOW": ("Ags", "Feeder Cattle"),
    "LEANHOG": ("Ags", "Lean Hogs"),
    "SUGAR11": ("Ags", "Sugar #11"),
    "COFFEE": ("Ags", "Coffee"),
    "COCOA": ("Ags", "Cocoa"),
    "COTTON2": ("Ags", "Cotton #2"),
}


def _get(path: str) -> pd.DataFrame:
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=120) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def _daily(df: pd.DataFrame, col: str) -> pd.Series:
    s = df.set_index(pd.to_datetime(df["DATETIME"]))[col].dropna()
    return s.groupby(s.index.normalize()).last()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _get("csvconfig/instrumentconfig.csv").set_index("Instrument")
    spread = _get("csvconfig/spreadcosts.csv").set_index("Instrument")["SpreadCost"]
    roll = _get("csvconfig/rollconfig.csv").set_index("Instrument")["HoldRollCycle"]

    frames, meta = [], []
    for code, (asset_class, name) in UNIVERSE.items():
        print("fetching", code)
        adj = _daily(_get(f"adjusted_prices_csv/{code}.csv"), "price")
        cur = _daily(_get(f"multiple_prices_csv/{code}.csv"), "PRICE")
        frames.append(pd.DataFrame({"instrument": code, "adj": adj, "price": cur.reindex(adj.index).ffill()}))
        meta.append({
            "instrument": code, "name": name, "asset_class": asset_class,
            "multiplier": float(cfg.loc[code, "Pointsize"]),
            "commission": float(cfg.loc[code, "PerBlock"]),
            "spread": float(spread.get(code, 0.0)),
            "rolls_per_year": len(str(roll.get(code, "HMUZ"))),
        })
    prices = pd.concat(frames).rename_axis("date").reset_index()
    prices.to_parquet(OUT / "adjusted_daily.parquet", index=False)
    pd.DataFrame(meta).to_csv(OUT / "instruments.csv", index=False)
    print(f"wrote {len(prices):,} rows for {len(meta)} instruments to {OUT}")


if __name__ == "__main__":
    main()
