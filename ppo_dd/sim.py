"""Bar-by-bar port of tradingview/ppo_dd.pine (one side at a time).

Mimics TradingView's broker emulator closely enough to rank parameter sets:
  * The script runs at each bar's close; market orders (base entry, DD add,
    TP catch-up close) fill at the NEXT bar's open.
  * The strategy.exit placed at bar i's close is live during bar i+1 only if a
    position existed at bar i's close (so the base-fill bar has no resting
    TP, which is why the Pine script has the TP catch-up).
  * Limit/stop fill on touch; a gap through the level fills at the open. When
    both TP and SL are inside one bar, TV's OHLC path rule decides: if the
    high is closer to the open than the low, the path is O-H-L-C, else O-L-H-C.
  * Pending DD fills at the open BEFORE the resting exit is checked, and the
    exit then covers the whole (enlarged) position at the levels computed on
    the previous bar, as in TV.

Inputs are plain numpy arrays so the core loop can be numba-compiled.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - slow fallback
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

# Result vector layout.
R_NET, R_MAXDD, R_TRADES, R_WINS, R_SL, R_MAXQTY, R_GROSS_W, R_GROSS_L, R_MAXDD_COUNT = range(9)
N_RESULTS = 9


@njit(cache=True)
def simulate(o, h, l, c, t_ms, day_id, ema1, ema2,
             direction, thr1, thr2, entry_qty, base_tp, max_dd, sl_perc, sl_at_max, tp_catch,
             dd_dist, dd_tp, mults, decay_min, min_tp, one_day,
             point_value, commission, slip, start_idx, end_idx, trade_log):
    """Return a results vector (see R_* indexes). Prices in points, P&L in $.

    trade_log: (k, 6) float array or a (0, 6) one to skip logging; columns =
    exit bar idx, side, qty, avg, exit price, net $.
    """
    res = np.zeros(N_RESULTS)
    d = direction
    pos = 0              # signed contracts at end of last processed bar
    avg = 0.0
    first_entry = np.nan
    total_usd = 0.0
    count_max = 0
    last_fill_t = 0
    pend_qty = 0         # market add/entry to fill at next open
    pend_close = False   # catch-up close at next open
    ex_live = False      # resting exit for the next bar
    ex_tp = np.nan
    ex_sl = np.nan
    last_entry_day = -1
    realized = 0.0
    peak = 0.0
    max_dd_usd = 0.0
    n_log = 0

    for i in range(start_idx, end_idx):
        prev_pos = pos
        closed_px = np.nan
        is_sl = False

        # ---- fills at the open ----
        if pend_close and pos != 0:
            closed_px = o[i] - d * slip
            pend_close = False
            ex_live = False
        if pend_qty > 0 and np.isnan(closed_px):
            px = o[i] + d * slip
            q = abs(pos)
            avg = (avg * q + px * pend_qty) / (q + pend_qty)
            pos = d * (q + pend_qty)
        pend_qty = 0
        pend_close = False

        # ---- resting TP/SL during the bar ----
        if ex_live and pos != 0 and np.isnan(closed_px):
            tp = ex_tp
            sl = ex_sl
            if d == 1:
                if o[i] >= tp:
                    closed_px = o[i]
                elif not np.isnan(sl) and o[i] <= sl:
                    closed_px = o[i] - slip
                    is_sl = True
                else:
                    hit_tp = h[i] >= tp
                    hit_sl = (not np.isnan(sl)) and l[i] <= sl
                    if hit_tp and hit_sl:
                        if h[i] - o[i] <= o[i] - l[i]:
                            closed_px = tp
                        else:
                            closed_px = sl - slip
                            is_sl = True
                    elif hit_tp:
                        closed_px = tp
                    elif hit_sl:
                        closed_px = sl - slip
                        is_sl = True
            else:
                if o[i] <= tp:
                    closed_px = o[i]
                elif not np.isnan(sl) and o[i] >= sl:
                    closed_px = o[i] + slip
                    is_sl = True
                else:
                    hit_tp = l[i] <= tp
                    hit_sl = (not np.isnan(sl)) and h[i] >= sl
                    if hit_tp and hit_sl:
                        if o[i] - l[i] <= h[i] - o[i]:
                            closed_px = tp
                        else:
                            closed_px = sl + slip
                            is_sl = True
                    elif hit_tp:
                        closed_px = tp
                    elif hit_sl:
                        closed_px = sl + slip
                        is_sl = True
        ex_live = False

        # worst intrabar open P&L for drawdown (before any close this bar)
        if pos != 0:
            worst = (l[i] if d == 1 else h[i])
            if not np.isnan(closed_px):
                worst = closed_px if (d == 1 and closed_px < worst) or (d == -1 and closed_px > worst) else worst
            eq_low = realized + (worst - avg) * d * abs(pos) * point_value
            if peak - eq_low > max_dd_usd:
                max_dd_usd = peak - eq_low

        if not np.isnan(closed_px):
            q = abs(pos)
            n_entries = count_max + 1
            pnl = (closed_px - avg) * d * q * point_value - commission * 2 * q
            realized += pnl
            res[R_TRADES] += 1
            if pnl > 0:
                res[R_WINS] += 1
                res[R_GROSS_W] += pnl
            else:
                res[R_GROSS_L] -= pnl
            if is_sl:
                res[R_SL] += 1
            if count_max > res[R_MAXDD_COUNT]:
                res[R_MAXDD_COUNT] = count_max
            if n_log < trade_log.shape[0]:
                trade_log[n_log, 0] = i
                trade_log[n_log, 1] = d
                trade_log[n_log, 2] = q
                trade_log[n_log, 3] = avg
                trade_log[n_log, 4] = closed_px
                trade_log[n_log, 5] = pnl
                n_log += 1
            pos = 0
            avg = 0.0

        if abs(pos) > res[R_MAXQTY]:
            res[R_MAXQTY] = abs(pos)

        # ---- script at bar close ----
        filled = pos != prev_pos
        if pos == 0:
            first_entry = np.nan
            total_usd = 0.0
            count_max = 0
            last_fill_t = 0
            if peak < realized:
                peak = realized
            ok_day = (not one_day) or last_entry_day != day_id[i]
            if ok_day and i + 1 < end_idx:
                if d == 1:
                    sig = ema1[i] / c[i] > thr1 and ema2[i] / c[i] > thr2
                else:
                    sig = c[i] / ema1[i] > thr1 and c[i] / ema2[i] > thr2
                if sig:
                    pend_qty = entry_qty
                    last_entry_day = day_id[i]
            continue

        if np.isnan(first_entry):
            first_entry = avg
            total_usd = abs(pos) * avg
        if filled:
            last_fill_t = t_ms[i]

        di = min(count_max, 10)
        stage_tp = base_tp if count_max == 0 else dd_tp[count_max - 1]
        mtp = min(min_tp[di], stage_tp)
        floor_ = avg * (1 + d * mtp)
        tp_full = (first_entry if count_max == 0 else avg) * (1 + d * stage_tp)
        live_tp = tp_full
        if decay_min[di] > 0 and last_fill_t != 0:
            el = (t_ms[i] - last_fill_t) / 60000.0
            f = min(1.0, max(0.0, el / decay_min[di]))
            live_tp = tp_full - (tp_full - floor_) * f

        armed = (count_max >= max_dd) if sl_at_max else True
        live_sl = avg * (1 - d * sl_perc) if armed else np.nan

        next_dd = np.nan
        if count_max < max_dd and dd_dist[count_max] > 0:
            next_dd = first_entry * (1 - d * dd_dist[count_max])

        catch = False
        if tp_catch and filled:
            touched = h[i] >= tp_full if d == 1 else l[i] <= tp_full
            inside = c[i] < tp_full if d == 1 else c[i] > tp_full
            if touched and inside:
                pend_close = True
                catch = True

        if not catch:
            ex_live = True
            ex_tp = live_tp
            ex_sl = live_sl
            if not np.isnan(next_dd):
                hit = l[i] <= next_dd if d == 1 else h[i] >= next_dd
                if hit:
                    new_cnt = count_max + 1
                    usd = total_usd * mults[new_cnt - 1]
                    add = max(1, int(np.round(usd / next_dd)))
                    pend_qty = add
                    count_max = new_cnt
                    total_usd += add * next_dd

    # mark any open position to the last close
    if pos != 0:
        realized += (c[end_idx - 1] - avg) * d * abs(pos) * point_value - commission * 2 * abs(pos)
    res[R_NET] = realized
    res[R_MAXDD] = max_dd_usd
    return res
