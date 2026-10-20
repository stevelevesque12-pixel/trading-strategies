"""Write a Tradeify 50K Select Flex ledger (summary, account events, payouts, every trade) to Excel.

Usage:
  python -m morning_dip.prop_ledger OUT.xlsx --eval-mnq 5 --funded-mnq 6 \
      --author-trades nq_3min_origin000s.csv [--tv-trades tradingview_export.csv] \
      [--backtest-trades trades_offset0.csv --year 2025]

Each source becomes its own scenario sheet set. Author trades are the 1-second trade lists the
strategy author shared (NQ, gross in $ per NQ, mae in points); TradingView trades are a
Strategy Tester "List of Trades" export (times in New York, prices on MNQ); backtest trades
come from morning_dip.backtest_1m.
"""
import argparse

import pandas as pd

from morning_dip.prop_sim import Rules, day_trades, simulate

CT = 'America/Chicago'


def author_trades(path):
    a = pd.read_csv(path).sort_values('entry_t_ct')
    return pd.DataFrame({'date': a.date, 'entry_time': a.entry_t_ct.str[11:], 'exit_time': a.exit_t_ct.str[11:],
                         'entry': a.level, 'exit': a.exit, 'exit_reason': a.reason_label,
                         'pts': a.gross / 20, 'mae': a.mae, 'source': a.source})


def tradingview_trades(path, after=None):
    t = pd.read_csv(path)
    en = t[t.Type.str.startswith('Entry')].set_index('Trade number')
    ex = t[t.Type.str.startswith('Exit')].set_index('Trade number')
    ent = pd.to_datetime(en['Date and time']).dt.tz_localize('America/New_York').dt.tz_convert(CT)
    ext = pd.to_datetime(ex['Date and time']).dt.tz_localize('America/New_York').dt.tz_convert(CT)
    out = pd.DataFrame({'date': ent.dt.strftime('%Y-%m-%d'), 'entry_time': ent.dt.strftime('%H:%M:%S'),
                        'exit_time': ext.dt.strftime('%H:%M:%S'), 'entry': en['Price USD'], 'exit': ex['Price USD'],
                        'exit_reason': ex.Signal.replace({'Dip Exit': 'target/stop'}),
                        'pts': ex['Price USD'] - en['Price USD'],
                        # Adverse excursion is per MNQ after half the round-trip cost.
                        'mae': ((-ex['Adverse excursion USD'] - ex['Commission USD'] / 2) / 2).clip(lower=0),
                        'source': 'TradingView MNQ 1s'})
    return out if after is None else out[out.date > after]


def backtest_trades(path, year=None):
    t = pd.read_csv(path)
    t = t[t.exit_i >= 0].sort_values('entry_t')
    clock = lambda s: pd.to_datetime(s, unit='s', utc=True).dt.tz_convert(CT).dt.strftime('%H:%M:%S')
    out = pd.DataFrame({'date': t.date, 'entry_time': clock(t.entry_t), 'exit_time': clock(t.exit_t),
                        'entry': t.level, 'exit': t.exit, 'exit_reason': t.reason_label,
                        'pts': t.gross / 20, 'mae': t.mae, 'source': 'NQ 1m backtest'})
    return out if year is None else out[out.date.str[:4] == str(year)]


def scenario(trades, sessions, eval_mnq, funded_mnq, r=Rules()):
    trades = trades.reset_index(drop=True)
    ledger = []
    summary, log = simulate(sessions, day_trades(trades, eval_mnq), r, day_trades(trades, funded_mnq), ledger)
    taken = pd.DataFrame(ledger)
    detail = trades.iloc[taken.row].reset_index(drop=True)
    detail = pd.concat([detail.drop(columns='mae'), taken.drop(columns=['day', 'row'])], axis=1)
    detail.insert(detail.columns.get_loc('pnl'), 'mnq', detail.phase.map({'eval': eval_mnq, 'funded': funded_mnq}))
    detail[['pnl', 'balance', 'floor']] = detail[['pnl', 'balance', 'floor']].round(2)
    events = pd.DataFrame(log, columns=['date', 'event'])
    pays = events[events.event.str.startswith('payout')].copy()
    pays['gross_payout'] = pays.event.str.extract(r'payout (\d+)').astype(float)
    pays['to_you'] = (pays.gross_payout * r.split).round(2)
    s = {'Period': f'{sessions[0]} to {sessions[-1]} ({len(sessions)} sessions)',
         'Size': f'{eval_mnq} MNQ in evaluation, {funded_mnq} MNQ funded',
         'Trades taken': len(detail),
         'Evaluations bought': summary['evals'],
         'Evaluations passed': summary['passes'],
         'Evaluations failed': int((events.event.str.startswith('eval account')).sum()),
         'Funded accounts lost': summary['funded_busts'],
         'Payouts': summary['payouts'],
         'Paid to you (90%)': round(summary['paid'], 2),
         'Evaluation fees': summary['fees'],
         'Net cash': round(summary['net_cash'], 2),
         'Open at the end': summary['end'],
         'Unwithdrawn funded profit at the end': round(summary['funded_open_profit'], 2)}
    return s, events, pays, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--eval-mnq', type=int, default=5)
    ap.add_argument('--funded-mnq', type=int, default=6)
    ap.add_argument('--eval-fee', type=float, default=100.)
    ap.add_argument('--author-trades')
    ap.add_argument('--tv-trades')
    ap.add_argument('--backtest-trades')
    ap.add_argument('--year')
    a = ap.parse_args()
    r = Rules(eval_fee=a.eval_fee)
    scenarios = {}
    if a.author_trades:
        t = author_trades(a.author_trades)
        if a.tv_trades:
            t = pd.concat([t, tradingview_trades(a.tv_trades, after=t.date.max())])
        scenarios['2026'] = t
    if a.backtest_trades:
        scenarios[str(a.year or 'backtest')] = backtest_trades(a.backtest_trades, a.year)
    rows, sheets = {}, {}
    for name, t in scenarios.items():
        sessions = sorted(t.date.unique())
        s, events, pays, detail = scenario(t, sessions, a.eval_mnq, a.funded_mnq, r)
        rows[name] = s
        sheets.update({f'{name} events': events, f'{name} payouts': pays, f'{name} trades': detail})
    with pd.ExcelWriter(a.out, engine='openpyxl') as xl:
        pd.DataFrame(rows).to_excel(xl, sheet_name='Summary')
        for sheet, frame in sheets.items():
            frame.to_excel(xl, sheet_name=sheet, index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(len(str(c.value or '')) for c in col) + 2)
    print(pd.DataFrame(rows).to_string())


if __name__ == '__main__':
    main()
