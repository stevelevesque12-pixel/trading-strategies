# Notes for Claude sessions

## Datasets (in `sample_data/`)

- `real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet`: **3 years of 1-minute NQ**
  (764 RTH sessions). Index is UTC with **bar-open** timestamps; convert with
  `pd.read_parquet(p).tz_convert("America/New_York")`. Columns are open/high/low/close/volume.
  It's a continuous contract and looks back-adjusted (absolute prices run high; point
  differences are fine). The source CSV hit Excel's row limit, so it may have been cut off at Dec 11 2025.
- `real_multi_instrument/real_*_{1m,5m,15m}_*.parquet`: micro futures (MNQ, MES, MYM, M2K, MGC,
  MCL, SIL), 1m Aug 2026, 5m May–Aug 2026, 15m Sep 2025–Aug 2026. Also NQ, ES, GC and SI 15m for 2016–2026.
- `daily/nasdaq_index_daily_2026-09-08_2026-10-07.csv`: daily Nasdaq index OHLC with no volume (22 days).

The same NQ 1m data is in `stevelevesque12-pixel/AI-personal-hedge-fund` at
`data/historical/raw/NQ_1m_2022-12-26_2025-12-11.csv.gz`.

## polyweather (Polymarket temperature markets)

- `python -m polyweather validate|backtest|scan`; see `polyweather/README.md` for results.
- Backtests are only meaningful with strict timing (default `obs_lag_min=5`) and
  `--fills trades` (default). Looser settings produced large fake edges.
- Open-Meteo's free quota is per IP and is often spent in shared cloud sandboxes; NBM/obs (IEM, NOAA) need no key.
  Set `POLYWEATHER_CACHE` to a scratch dir so archived responses are reused across runs.
