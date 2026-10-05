"""Numba bar-loop backtester: ATR-band bracket exits, regime-scaled risk sizing, fees, slippage.

Execution model (deliberately conservative):
  * Signal evaluated on bar i's close -> market entry at bar i+1's open, +slippage.
  * Stop / target are ATR bands set from the fill: fill -/+ sl_k*ATR, fill +/- tp_k*ATR
    (ATR as of the signal bar, so no lookahead).
  * Stops fill at the stop price -slippage (or at the open, -slippage, if the bar gaps
    through it). Targets are resting limits: fill at the target, no slippage.
  * Intrabar order (v5): the extreme nearer the open is assumed to trade first (TradingView's
    broker-emulator rule), so a bar can hit its stop or its target first depending on shape.
  * Positions are flattened at the session cutoff (market, -slippage) and never held
    across sessions.
  * Fees are charged per contract round turn; slippage is charged in ticks per side.
  * Optional trailing ATR stop (trail_k > 0): the stop ratchets to the best price since
    entry -/+ trail_k*ATR *intrabar* (like a broker trailing stop), so a bar that rallies
    and then reverses can stop out on the same bar; the ATR target stays as a cap.
  * Optional self-imposed daily loss limit / daily profit cap stop new entries for the
    rest of that session (prop-firm risk + consistency-rule management).
"""

import numpy as np
from numba import njit

# columns of the trade matrix returned by run()
T_ENTRY_I, T_EXIT_I, T_DIR, T_ENTRY_PX, T_EXIT_PX, T_QTY, T_PNL, T_MAE, T_RISK, T_REASON, T_REGIME = range(11)
N_COLS = 11
REASONS = {0: "stop", 1: "target", 2: "flatten", 3: "trend_flip"}


@njit(cache=True)
def run(o, h, l, c, atr_v, long_sig, short_sig, trend_dir, regime, can_enter, flatten, day_id,
        sl_k, tp_k, risk_usd, small_mult, pv, tick, fee_rt, slip_ticks, max_qty,
        daily_loss_limit, daily_profit_cap, exit_on_flip, trail_k):
    n = len(c)
    out = np.zeros((n // 2 + 1, N_COLS))
    nt = 0
    slip = slip_ticks * tick

    pos = 0  # +1 long, -1 short
    qty = 0
    entry_px = 0.0
    stop = 0.0
    target = 0.0
    entry_i = 0
    mae = 0.0
    risk_amt = 0.0
    reg = 0.0
    pos_atr = 0.0
    extreme = 0.0

    pending = 0
    pend_qty = 0
    pend_atr = 0.0
    pend_reg = 0.0

    cur_day = -1
    day_pnl = 0.0
    locked = False

    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            day_pnl = 0.0
            locked = False

        # ---- fill pending entry at this bar's open
        if pending != 0:
            if day_id[i] == day_id[i - 1] and not flatten[i - 1]:
                pos = pending
                qty = pend_qty
                entry_px = o[i] + pos * slip
                stop = entry_px - pos * sl_k * pend_atr
                target = entry_px + pos * tp_k * pend_atr
                entry_i = i
                mae = 0.0
                risk_amt = sl_k * pend_atr * pv * qty
                reg = pend_reg
                pos_atr = pend_atr
                extreme = pos * entry_px  # best price since entry, in favourable coordinates
            pending = 0

        # ---- manage open position on this bar
        # Intrabar path, same assumption as TradingView's broker emulator: the extreme nearer
        # the open is visited first (O-H-L-C or O-L-H-C). Work in "favourable" coordinates
        # x = pos * price so long and short share one code path.
        if pos != 0:
            exit_px = 0.0
            reason = -1
            fav = h[i] if pos > 0 else l[i]
            adv = l[i] if pos > 0 else h[i]
            xo, xf, xa, xc = pos * o[i], pos * fav, pos * adv, pos * c[i]
            xs, xt = pos * stop, pos * target
            worst = xa
            if xo <= xs:
                reason, xexit = 0, xo
                worst = xo
            elif xo >= xt:
                reason, xexit = 1, xo
            elif xf - xo < xo - xa:  # favourable extreme first
                if xf >= xt:
                    reason, xexit = 1, xt
                else:
                    if trail_k > 0 and xf > extreme:
                        extreme = xf
                        xs = max(xs, extreme - trail_k * pos_atr)
                    if xa <= xs:
                        reason, xexit = 0, xs
                        worst = xs
            else:  # adverse extreme first
                if xa <= xs:
                    reason, xexit = 0, xs
                    worst = xs
                elif xf >= xt:
                    reason, xexit = 1, xt
                else:
                    if trail_k > 0 and xf > extreme:
                        extreme = xf
                        xs = max(xs, extreme - trail_k * pos_atr)
                    if xc <= xs:
                        reason, xexit = 0, xs
            stop = xs / pos
            if reason == 0:
                exit_px = xexit / pos - pos * slip
            elif reason == 1:
                exit_px = xexit / pos
            adverse = (worst - pos * entry_px) * pv * qty
            if adverse < mae:
                mae = adverse
            if reason < 0 and flatten[i]:
                exit_px, reason = c[i] - pos * slip, 2
            if reason < 0 and exit_on_flip and trend_dir[i] == -pos:
                exit_px, reason = c[i] - pos * slip, 3
            if reason >= 0:
                fees = fee_rt * qty
                pnl = (exit_px - entry_px) * pos * pv * qty - fees
                out[nt, T_ENTRY_I] = entry_i
                out[nt, T_EXIT_I] = i
                out[nt, T_DIR] = pos
                out[nt, T_ENTRY_PX] = entry_px
                out[nt, T_EXIT_PX] = exit_px
                out[nt, T_QTY] = qty
                out[nt, T_PNL] = pnl
                out[nt, T_MAE] = min(mae - fees, pnl)
                out[nt, T_RISK] = risk_amt
                out[nt, T_REASON] = reason
                out[nt, T_REGIME] = reg
                nt += 1
                day_pnl += pnl
                if day_pnl <= -daily_loss_limit or day_pnl >= daily_profit_cap:
                    locked = True
                pos = 0
                # exit bar can't also be a signal bar for a same-bar re-entry fill
                continue
        # ---- new signal on this bar's close
        if pos == 0 and not locked and can_enter[i] and not flatten[i] and i + 1 < n:
            d = 0
            if long_sig[i]:
                d = 1
            elif short_sig[i]:
                d = -1
            if d != 0 and regime[i] > 0 and atr_v[i] > 0:
                mult = 1.0 if regime[i] >= 2 else small_mult
                per_ct = sl_k * atr_v[i] * pv + fee_rt + 2 * slip * pv
                q = int(risk_usd * mult / per_ct)
                if q > max_qty:
                    q = max_qty
                if q >= 1:
                    pending = d
                    pend_qty = q
                    pend_atr = atr_v[i]
                    pend_reg = regime[i]
    return out[:nt]
