# Failed 2s (TheStrat) — intraday automation for Tradovate

Codifies the "Failed 2s" multi-timeframe reversal system (Failed-2 bias +
Market Structure Shift/FVG entry trigger) popularized in TheStrat community,
with a backtester and a Tradovate live-execution layer.

**This was built from research, not from watching the source video** — the
X/Twitter video this was requested from couldn't be accessed from this
environment (egress to x.com is blocked here). The rules below are a
best-effort reconstruction from public TheStrat/Trader-Mike material (see
research summary earlier in this conversation). Validate against your own
understanding of the setup before trusting it with money.

**A second, higher-frequency strategy also lives in this repo**:
`structure_scalp/` -- 1-minute structure direction as bias, then on the 5s
chart: wait for a pullback against that bias, wait for the 5s to break back
in the bias direction (the reversal leg), anchor a VWAP to that leg's start,
and enter on the retest touch of that VWAP. Stop = the leg's low/high,
target = a fixed R multiple. See `structure_scalp/strategy.py` and
`tradingview/structure_scalp.pine` (works for MES or MNQ -- same tick size,
just set the point-value input per symbol). It reuses the swing/MSS
primitives below but is a different strategy from Failed-2s, built because
Failed-2s trades too infrequently to be practical for a prop-firm payout
timeline. An earlier version of this strategy entered immediately on every
5s structure break while 15m+1m aligned; that traded more often but mostly
on 5s noise, so it was replaced with the pullback+VWAP-retest sequence
above for better trade quality. Not backtested against real data -- no free
source available provides sub-1-minute history, so it's logic-tested
against synthetic bars only (`tests/test_structure_scalp.py`); forward-test
carefully before trusting it.

## Strategy logic (Failed-2s)

1. **Bias timeframe** — a Failed-2 (`F2U`/`F2D`) completes: a directional (2)
   candle that reverses and closes back against its own break (a failed
   liquidity sweep). This sets a directional bias and the swept level.
2. **Entry timeframe** — while that bias is active and we're inside the
   intraday entry window, wait for a Market Structure Shift (a strong-bodied
   close through the last confirmed opposite swing point) in the bias
   direction, confirmed by a Fair Value Gap. That's the entry trigger.
3. **Stop** — beyond the tighter of the broken swing point or the bias
   timeframe's swept level, plus a small tick buffer.
4. **Target** — `target_r * risk` (default 1:1).
5. **Intraday only** — all state (bias, swing structure) resets every session
   day. No new entries after `no_entry_after` (default 15:45 ET); any open
   position is force-flattened at `flatten_at` (default 15:55 ET).

Timeframe pairs implemented (entry timeframe, bias timeframe):

| Pair | Entry TF | Bias TF |
|---|---|---|
| `1m-15m` | 1 minute | 15 minute |
| `5m-1h` | 5 minute | 1 hour |
| `15m-4h` | 15 minute | 4 hour |

Code layout:
- `failed2s/` — pure strategy logic (bar classification, Failed-2, swing/MSS/FVG, the signal engine, risk manager, instrument specs). Shared by backtest, live, and the webhook path.
- `backtest/` — event-driven backtester over OHLCV CSV data.
- `live/` — Tradovate REST/WebSocket client + a Python live runner (polls/streams Tradovate directly).
- `tradingview/` — the current recommended way to go live: a Pine Script port runs on TradingView (which has the market data and computes risk-based position size), fires a webhook formatted for **TradersPost** (a hosted bridge that connects to Tradovate with a regular login — no paid API Access Add-On needed, which matters on a prop-firm sim account). See **TRADINGVIEW_WEBHOOK.md** for setup.
- `webhook/` — a self-hosted alternative to TradersPost (`webhook/server.py` talks to Tradovate directly), kept for if/when direct Tradovate API credentials become available. See the "Alternative" section at the bottom of TRADINGVIEW_WEBHOOK.md.
- `tests/` — unit tests for the strategy logic, backtest smoke tests, and webhook sizing/routing tests.

## Setup

```bash
pip install -r requirements.txt
```

## Backtesting

Data format: a CSV with `timestamp,open,high,low,close[,volume]` at
**1-minute** resolution (all three pairs are resampled from this single base
resolution).

**Real historical data.** `sample_data/fetch_real_data.py` pulls genuine
historical 1-minute bars (OANDA S&P 500 / Nasdaq-100 index CFD data,
republished under GPL-3.0 by the `FutureSharks/financial-data` GitHub repo)
and converts them to this repo's CSV schema:

```bash
python sample_data/fetch_real_data.py --instrument SPX500_USD --start-year 2018 --end-year 2019 --out sample_data/real_spx500_2018_2019.csv
```

Read the caveats at the top of that script before trusting results: this is
an index CFD proxy (not literal CME ES/MES tick data), coverage only goes
through mid-2020, and volume is OANDA's tick count, not real exchange
volume. It's genuine market data though — good enough to see whether the
pattern has any real edge, not a substitute for validating against recent
CME futures data before going live. For that, see Databento or FirstRate
Data (both require a paid/API-key account not set up in this environment).

A synthetic-data generator is also included, but only for smoke-testing the
pipeline (pure random walk, no real edge signal to find):

```bash
python sample_data/generate_sample.py --days 20 --out sample_data/sample_1min.csv
```

**`sample_data/real_multi_instrument/`** — real CME/COMEX/CBOT/NYMEX
futures data supplied directly by the user (not via `fetch_real_data.py`/
`fetch_yfinance.py`), stored as parquet rather than CSV specifically so it
sidesteps `.gitignore`'s `sample_data/real_*.csv` rule (that rule exists
on purpose - real data is normally meant to be fetched on demand, not
committed - see the git history for the discussion before adding more
here). `--data` now accepts these directly; `backtest/data.py`'s
`load_1m_csv` dispatches on file extension.

- Micro contracts (MES, MNQ, MYM, MGC, MCL, SIL, M2K) at 1m/5m/15m -
  depth is per-interval, shared across all seven symbols: 1m covers
  2026-08-02 to 2026-08-25 (~3.5 weeks), 5m covers 2026-05-10 to
  2026-08-25 (~3.5 months), 15m covers 2025-09-30 to 2026-08-25
  (~11 months).
- Full-size contracts (ES, NQ, GC, SI) at 15m only, 2016-05 to 2026-08
  (~10 years, ~240k bars each).

Only the 1m files are genuine 1-minute data - the `1m-15m` pair needs
that resolution specifically. The 5m/15m files load and resample fine
for `5m-1h`/`15m-4h` (their OHLC is just built from coarser source bars
than a true 1-minute feed would give you), but don't point them at
`1m-15m` - the resample can't invent finer bars than the file has.

Provenance beyond "the user supplied these directly" wasn't stated and
hasn't been independently verified (which vendor, whether `volume` is
real exchange volume) - treat results from this data as "real market
data, source TBD" rather than production-grade until that's confirmed,
same caveat as the OANDA data above.

**Single pair:**

```bash
python -m backtest.run --data sample_data/sample_1min.csv --pair 5m-1h --symbol MES
```

**Compare all three pairs on the same data** (this is what you want to
figure out which pair performs best):

```bash
python -m backtest.compare --data sample_data/sample_1min.csv --symbol MES
```

Prints a side-by-side metrics table (num trades, win rate, total P&L,
profit factor, avg R, max drawdown, expectancy) and writes:
- `trades_<pair>.csv` — a full trade log per pair
- `pair_comparison.csv` — the summary table

Both CLIs support `--symbol` (MES, ES, MNQ, NQ, MCL, CL, MGC, GC),
`--contracts`, `--daily-loss-limit`, `--max-daily-trades`, `--target-r`.

**Backtest assumptions/limitations** (so you know what you're looking at):
fills are simulated at the triggering bar's close and stop/target hits are
checked on subsequent entry-timeframe bars' high/low (one bar of latency, no
slippage/commission modeled); if a bar's range hits both stop and target the
stop is assumed to fill first (conservative). 4-hour resampling floors to
UTC-clock-hour boundaries via pandas, which may not perfectly match how your
data provider aligns 4H candles. The engine force-closes any open position
the instant it sees a bar dated after the entry day (in addition to the
normal same-day flatten-at cutoff), so a position can never silently ride
across multiple sessions -- but if the data feed itself has no bars between
the cutoff and a holiday reopen (Thanksgiving, Christmas, etc.), the
position closes on the first bar that exists, which can be a day or two
later purely because there's no earlier price to close it at. This showed
up in the real SPX500 backtest (2 of 1330 trades, both on US holiday
weekends) -- worth checking your trade log for exit_reason=session_flatten
rows with a multi-day gap before relying on this unattended over holidays.

## Live trading on Tradovate

Three ways to go live:

- **TradingView + TradersPost** (`tradingview/failed2s_mes.pine`) — the
  current recommended path. TradingView runs the signal logic and computes
  risk-based position size directly in Pine, fires a webhook formatted for
  TradersPost, which connects to Tradovate with a regular login (no paid
  API Access Add-On needed). See **TRADINGVIEW_WEBHOOK.md**.
- **TradingView + self-hosted webhook** (`webhook/`) — same idea, but you
  host the bridge yourself instead of paying for TradersPost. Needs direct
  Tradovate API credentials (CID/Secret), which require the paid API
  Access Add-On on a live funded account — not available on most prop-firm
  sim accounts. See the "Alternative" section of TRADINGVIEW_WEBHOOK.md.
- **Python runner** (`live/runner.py`, below) — runs the same
  `failed2s/strategy.py` logic directly against Tradovate's own market data
  feed, no TradingView involved, fixed 1-contract sizing. Same API access
  requirement as the self-hosted webhook path.

**Read this before pointing any of these at a funded account.** The Tradovate client
(`live/tradovate_client.py`) follows Tradovate's public API docs but has
**not been tested end-to-end against a real Tradovate account** from this
environment (no credentials, no network access to tradovateapi.com here).
Validate every call against a **demo** account first.

Environment variables:

```bash
export TRADOVATE_USERNAME=...
export TRADOVATE_PASSWORD=...
export TRADOVATE_APP_ID=...
export TRADOVATE_CID=...        # from Tradovate API settings
export TRADOVATE_SEC=...        # from Tradovate API settings
export TRADOVATE_APP_VERSION=1.0   # optional
export TRADOVATE_DEVICE_ID=failed2s-bot  # optional
```

Dry run (default — logs signals, sends no orders):

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username>
```

Live (demo environment, explicit opt-in):

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username> --live --env demo
```

Only after demo validation, live/real money:

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username> --live --env live
```

Known gap: the runner logs each signal and places a bracket order but does
not yet wire Tradovate's order-fill/user-sync WebSocket back into the risk
manager, so the daily-loss-limit lockout won't see live fills until you add
that callback (`LiveRunner._on_finished_bar` has a note where to hook it
in). Don't rely on the daily loss limit unattended until that's wired up.

## Running the tests

```bash
pytest
```

## IB Break (Initial Balance breakout) -- `ib_break/`

The IB is the 09:30-10:30 ET high/low. `ib_break/stats.py` reproduces the
breach-rate study it's based on (10y NQ 15m, 2,633 days): either side breaks
by noon 82.6% / by close 96.0%; IB high 47.1% / 62.6%; IB low 40.0% / 54.8%.
Two confluences pick the side: IB closing above its midpoint, and the IB high
being printed *last* (low first). Both bullish -> IB high breaks 74.5% by noon /
83.9% by close (bearish mirror: low 67.8% / 77.9%).

**A breach is not a winning trade.** After the high breaks, price reaches
+0.5x IB range before returning to IB mid only ~45% of the time (the `follow`
columns), so a "breakout, stop at mid, 1:1" trade loses money: unfiltered
close-entry / mid-stop / 1R / noon cutoff on 3y of 1m NQ = 614 trades, PF 0.99,
-$5.8k net.

What does hold up (`CONFLUENCE_PRESET`, `--preset`):
- trade only the side both confluences agree on (no trade on conflicting days, ~16%)
- buy-stop at IB high / sell-stop at IB low, any time 10:30-15:45
- stop at the *opposite* IB extreme (full IB range of risk, median ~100 NQ pts)
- 2R target, which is rarely hit -- ~75% of trades exit at the 15:55 flatten,
  so in effect "take the confluence-side break, hold to the close"

| NQ, 1 contract, net of $4.50 RT + 1 tick/side | trades | win% | avg R | net PF | net P&L | max DD |
|---|---|---|---|---|---|---|
| 15m, 2016-05 .. 2026-08, preset | 1681 | 51.7 | +0.066 (t=3.3) | 1.20 | $217k | $54k |
| 15m, same, no confluence filter | 2512 | 49.9 | +0.035 (t=2.1) | 1.10 | $160k | $67k |
| 1m, 2022-12 .. 2025-12, preset | 516 | 52.9 | +0.071 (t=2.0) | 1.21 | $86k | $48k |
| 1m, same, no confluence filter | 736 | 51.0 | +0.047 (t=1.6) | 1.14 | $80k | $63k |

Positive in both directions (15m: longs +0.068R, shorts +0.062R) and in 9 of
11 calendar years (losers: 2020 and 2025). Caveats: the edge is thin (~0.07R
per trade) against a full-IB-range stop, so drawdowns are large -- $50k on one
NQ is ~$5k on one MNQ, which is already most of a typical prop-firm trailing
drawdown; size accordingly. The preset was chosen as the best of a 64-config
sweep (`--sweep`), but its whole neighbourhood (touch entry + opposite stop)
is positive on both datasets, and the 1m and 15m files overlap for 2023-2025
so they're not independent tests. Touch fills on 15m bars ignore a same-bar
stop-out (unlikely with a full-IB stop, but slightly optimistic).

```
python -m ib_break.stats --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet
python -m ib_break.run --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet --entry-tf 15 --preset
python -m ib_break.run --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet --sweep
```

TradingView port: `tradingview/ib_break.pine` (preset defaults, 1m or 5m
chart, NQ or MNQ). It sends a resting stop-entry order with a bracket to
TradersPost at 10:30, a cancel at 15:45 if unfilled, and an exit at 15:55.
Alerts are only sent for the single-direction filter modes. See the header
of the file and TRADINGVIEW_WEBHOOK.md.
