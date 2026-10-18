"""
Markout and P&L report from a paper-trading event log.

Markout at horizon h, per share, in cents, positive = good for us:
    buy:  mid(t + h) - fill price
    sell: fill price - mid(t + h)
At h=0 it's the edge captured vs. mid at fill time. If markouts decay to
negative at longer horizons, the people trading against us know more than
we do, and spread + rewards has to more than pay for that.

Usage: python -m polymarket_mm.report runs/paper_YYYYMMDD_HHMMSS.jsonl
"""

import argparse
import json

import pandas as pd

HORIZONS = (0, 10, 60, 300, 1800)


def load(path) -> dict[str, pd.DataFrame]:
    recs: dict[str, list] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault(r["type"], []).append(r)
    return {k: pd.DataFrame(v) for k, v in recs.items()}


def markouts(fills: pd.DataFrame, mids: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Adds mo_{h}s columns (cents/share). NaN when the log ends before t+h."""
    out = []
    for mkt, f in fills.groupby("market"):
        m = mids[mids["market"] == mkt].sort_values("ts")
        f = f.sort_values("ts").copy()
        last_ts = m["ts"].max() if len(m) else float("-inf")
        sign = f["side"].map({"BUY": 1, "SELL": -1})
        for h in horizons:
            probe = pd.DataFrame({"ts": f["ts"] + h})
            if len(m):
                got = pd.merge_asof(probe, m[["ts", "mid"]], on="ts", direction="backward")["mid"].values
            else:
                got = float("nan")
            mo = sign * (pd.Series(got, index=f.index) - f["price"]) * 100
            mo[f["ts"] + h > last_ts] = float("nan")
            f[f"mo_{h}s"] = mo
        out.append(f)
    return pd.concat(out) if out else fills


def summary(logs: dict[str, pd.DataFrame], horizons=HORIZONS) -> pd.DataFrame:
    fills = logs.get("fill", pd.DataFrame(columns=["market", "ts", "side", "price", "size", "fee"]))
    mids = logs.get("mid", pd.DataFrame())
    rewards = logs.get("reward", pd.DataFrame())
    meta = logs.get("meta", pd.DataFrame())
    names = dict(zip(meta["market"], meta["question"])) if len(meta) else {}
    markets = set(mids["market"]) if len(mids) else set()
    rows = []
    mo = markouts(fills, mids, horizons) if len(fills) else fills
    for mkt in sorted(markets):
        f = mo[mo["market"] == mkt] if len(mo) else mo
        last_mid = mids[mids["market"] == mkt].sort_values("ts")["mid"].iloc[-1]
        signed = (f["side"].map({"BUY": 1, "SELL": -1}) * f["size"]) if len(f) else pd.Series(dtype=float)
        cash = -(signed * f["price"]).sum() - (f["fee"].sum() if len(f) else 0.0)
        pos = signed.sum()
        rw = rewards[rewards["market"] == mkt]["amount"].sum() if len(rewards) else 0.0
        row = {
            "market": names.get(mkt, mkt)[:50],
            "fills": len(f),
            "shares": f["size"].sum() if len(f) else 0.0,
            "position": pos,
            "trading_pnl": cash + pos * last_mid,
            "est_rewards": rw,
        }
        row["total"] = row["trading_pnl"] + rw
        for h in horizons:
            col = f"mo_{h}s"
            ok = f[f[col].notna()] if len(f) else f
            row[f"mo_{h}s_c"] = (ok[col] * ok["size"]).sum() / ok["size"].sum() if len(ok) else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        total = {"market": "TOTAL", **{c: df[c].sum() for c in
                 ("fills", "shares", "position", "trading_pnl", "est_rewards", "total")}}
        if len(mo):
            for h in horizons:
                col = f"mo_{h}s"
                ok = mo[mo[col].notna()]
                total[f"mo_{h}s_c"] = (ok[col] * ok["size"]).sum() / ok["size"].sum() if len(ok) else float("nan")
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df


def format_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "no data logged"
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        return df.to_string(index=False, float_format=lambda x: f"{x:,.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--fills-csv", help="also write fills with markout columns to this CSV")
    a = ap.parse_args(argv)
    logs = load(a.log)
    print(format_summary(summary(logs)))
    print("\nmo_Xs_c = size-weighted markout X seconds after the fill, cents/share (+ = good). "
          "est_rewards is an estimate; see polymarket_mm/rewards.py.")
    if a.fills_csv and "fill" in logs:
        markouts(logs["fill"], logs["mid"]).to_csv(a.fills_csv, index=False)


if __name__ == "__main__":
    main()
