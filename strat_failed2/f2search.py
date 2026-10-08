"""Exhaustive 1-3 filter search over failed 2 signals. Train = 2023-2024, test = 2025 (never used to select)."""
import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2025-01-01", tz="America/New_York")


def conditions(t):
    c = {
        "long": t.dir == 1, "short": t.dir == -1,
        "rib_align": t.rib_align, "rib_strict": t.rib_strict, "rib_close": t.rib_close, "rib_against": t.rib_against,
        "ema50": t.ema50, "!ema50": ~t.ema50,
        "htf120": t.htf120, "!htf120": ~t.htf120, "htf480": t.htf480, "!htf480": ~t.htf480, "htf_slope": t.htf_slope,
        "vwap_rth": t.vwap_rth, "!vwap_rth": ~t.vwap_rth, "vwap_eth": t.vwap_eth, "!vwap_eth": ~t.vwap_eth,
        "ftfc_day": t.ftfc_day, "!ftfc_day": ~t.ftfc_day, "ftfc_60": t.ftfc_60, "!ftfc_60": ~t.ftfc_60,
        "ftfc_wk": t.ftfc_wk, "ftfc_pdc": t.ftfc_pdc, "daily_up": t.daily_up, "!daily_up": ~t.daily_up,
        "sw_lod": t.sw_lod, "sw_on": t.sw_on, "sw_pd": t.sw_pd, "sw_or": t.sw_or, "sw_any": t.sw_any, "!sw_any": ~t.sw_any,
        "loc<.25": t.loc_day < .25, "loc>.6": t.loc_day > .6,
        "risk_atr<.5": t.risk_atr < .5, "risk_atr.5-1": t.risk_atr.between(.5, 1), "risk_atr>1": t.risk_atr > 1,
        "rng_atr>1.2": t.rng_atr > 1.2, "rng_atr<.8": t.rng_atr < .8,
        "risk_adr<.04": t.risk_adr < .04, "risk_adr>.08": t.risk_adr > .08,
        "close_loc>.75": t.close_loc > .75, "body>.4": t.body > .4, "body<.2": t.body < .2,
        "relvol>1.3": t.relvol > 1.3, "relvol<.8": t.relvol < .8,
        "sweep>8t": t.sweep_ticks > 8, "sweep<=4t": t.sweep_ticks <= 4,
        "prev_same": t.prev_same, "prev_opp": t.prev_opp, "prev_in": t.prev_in,
        "t<1030": t.tod < 630, "t<1130": t.tod < 690, "t1030-14": t.tod.between(630, 839), "t>=14": t.tod >= 840,
        "!lunch": (t.tod < 690) | (t.tod >= 810),
    }
    return {k: np.asarray(v, bool) for k, v in c.items()}


def search(t, tm, min_tr=120, depth=3):
    """Return dataframe of filter combos with train stats (independent trades)."""
    tr = t[t.time < TRAIN_END]
    C = conditions(tr); names = list(C); M = np.stack([C[k] for k in names], 1).astype(np.float64)
    p = tr[f"p{tm}"].to_numpy()
    y = tr.time.dt.year.to_numpy()
    W = np.stack([np.ones_like(p), np.clip(p, 0, None), np.clip(-p, 0, None), p * p,
                  np.clip(p, 0, None) * (y == 2023), np.clip(-p, 0, None) * (y == 2023),
                  np.clip(p, 0, None) * (y == 2024), np.clip(-p, 0, None) * (y == 2024)], 1)
    K = len(names); rows = []
    firsts = [None] + list(range(K)) if depth >= 3 else [None]
    for a in firsts:
        Ma = M if a is None else M * M[:, [a]]
        # pairs (b<=c) given a: sums over signals of Ma[:,b]*M[:,c]*W[:,q]
        S = np.stack([(Ma * W[:, [q]]).T @ M for q in range(W.shape[1])], 2)
        bi, ci = np.triu_indices(K)
        for b, c in zip(bi, ci):
            if a is not None and not (a < b < c):
                continue
            s = S[b, c]
            if s[0] < min_tr:
                continue
            rows.append((a, b, c, *s))
    r = pd.DataFrame(rows, columns=["a", "b", "c", "n", "w", "l", "ss", "w23", "l23", "w24", "l24"])
    r["pf"] = r.w / r.l
    r["pf23"] = r.w23 / r.l23; r["pf24"] = r.w24 / r.l24
    mean = (r.w - r.l) / r.n; var = r.ss / r.n - mean ** 2
    r["t"] = mean / np.sqrt(var / r.n)
    r["net"] = r.w - r.l
    lab = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else names[int(x)]
    r["filters"] = [" & ".join(sorted({lab(a), names[b], names[c]} - {""})) for a, b, c in zip(r.a, r.b, r.c)]
    return r.drop_duplicates("filters")


def apply(t, filters):
    C = conditions(t); m = np.ones(len(t), bool)
    for f in filters.split(" & "):
        m &= C[f]
    return t[m]
