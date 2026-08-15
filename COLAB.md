# Running real-data backtests from Google Colab

This sandboxed environment can't reach Yahoo Finance (or any market-data
vendor) directly, but Colab has normal internet access, so it can fetch
recent Yahoo Finance data and hand it back for a real-data backtest.

Yahoo's own intraday limits (`sample_data/fetch_yfinance.py` documents
these) mean the pairs need different source data:
- `1m-15m` needs true 1-minute bars -> Yahoo only gives the **last 7 days** at 1m
- `5m-1h` and `15m-4h` only need 5-minute bars -> Yahoo gives the **last 60 days** at 5m

## Colab cells

**1. Get the code.** If the repo is public:

```python
!git clone https://github.com/stevelevesque12-pixel/trading-strategies.git
%cd trading-strategies
!git checkout claude/prop-firm-automation-340um6
```

If it's private, use a GitHub personal access token instead of the plain URL:

```python
from getpass import getpass
token = getpass("GitHub token: ")
!git clone https://{token}@github.com/stevelevesque12-pixel/trading-strategies.git
%cd trading-strategies
!git checkout claude/prop-firm-automation-340um6
```

**2. Install dependencies:**

```python
!pip install -q -r requirements.txt yfinance
```

**3. Fetch real Yahoo Finance data** (swap `MES=F` for `ES=F`, `NQ=F`, `MNQ=F`, `CL=F`, `GC=F` as needed):

```python
!python sample_data/fetch_yfinance.py --symbol MES=F --interval 1m --period 7d --out sample_data/yf_mes_1m.csv
!python sample_data/fetch_yfinance.py --symbol MES=F --interval 5m --period 60d --out sample_data/yf_mes_5m.csv
```

**4. Run the backtests** (each data file only supports the pairs its resolution can build):

```python
!python -m backtest.compare --data sample_data/yf_mes_1m.csv --symbol MES --pairs 1m-15m \
    --out-prefix yf_trades --summary-out yf_pair_comparison_1m.csv

!python -m backtest.compare --data sample_data/yf_mes_5m.csv --symbol MES --pairs 5m-1h,15m-4h \
    --out-prefix yf_trades --summary-out yf_pair_comparison_5m.csv
```

**5. Download the results** to your machine, or send them back to me here to interpret:

```python
from google.colab import files
files.download("yf_pair_comparison_1m.csv")
files.download("yf_pair_comparison_5m.csv")
files.download("yf_trades_1m-15m.csv")
files.download("yf_trades_5m-1h.csv")
files.download("yf_trades_15m-4h.csv")
```

## What to expect

`MES=F`/`ES=F` on Yahoo are the **continuous front-month futures contract**
(unlike the OANDA index-CFD data used for the 2018-2019 backtest) -- this is
much closer to what you'll actually trade on Tradovate. The tradeoff is a
much smaller sample: 7 days of 1-minute data is roughly a week of trading,
not enough to draw strong statistical conclusions about the `1m-15m` pair
specifically -- treat it as a sanity check, not a verdict. The 60-day 5m
data for the other two pairs gives a more usable sample size.

## Lab Model (Trader Kane's NQ strategy)

The Lab Model needs **two** symbols (NQ traded, ES for SMT), so it's the
`--symbol=F` swap above plus a second fetch. Its best-known config after
cost-aware optimization is `exec_swing_strength=5` at 1-minute execution --
but 1-minute is capped at 7 days by Yahoo, same limit as above. For a
60-day window, use the 5-minute config found in the same sweep instead
(`stop_buffer_ticks=0, exec_swing_strength=3, breakeven_at_r=0.5`) -- see
README.md's "Re-optimizing after adding costs" section for where these came
from.

**1. Get the code** (same as step 1 above) and **install dependencies**
(same as step 2 above, `pip install -q -r requirements.txt yfinance`).

**2. Fetch both symbols.** For the 60-day/5-minute test:

```python
!python sample_data/fetch_yfinance.py --symbol NQ=F --interval 5m --period 60d --out sample_data/yf_nq_5m.csv
!python sample_data/fetch_yfinance.py --symbol ES=F --interval 5m --period 60d --out sample_data/yf_es_5m.csv
```

Optionally, also grab the 7-day 1-minute pair to sanity-check the
1-minute/`exec_swing_strength=5` config on its (much shorter) native window:

```python
!python sample_data/fetch_yfinance.py --symbol NQ=F --interval 1m --period 7d --out sample_data/yf_nq_1m.csv
!python sample_data/fetch_yfinance.py --symbol ES=F --interval 1m --period 7d --out sample_data/yf_es_1m.csv
```

**3. Run the backtest:**

```python
!python -m backtest.run_lab_model --nq-data sample_data/yf_nq_5m.csv --es-data sample_data/yf_es_5m.csv \
    --execution-tf 5min --stop-buffer-ticks 0 --exec-swing-strength 3 --breakeven-at-r 0.5 \
    --out yf_lab_model_trades_5m.csv

# if you also fetched the 1-minute pair:
!python -m backtest.run_lab_model --nq-data sample_data/yf_nq_1m.csv --es-data sample_data/yf_es_1m.csv \
    --out yf_lab_model_trades_1m.csv
```

**4. Download the results** (or send the printed metrics/CSVs back here to interpret):

```python
from google.colab import files
files.download("yf_lab_model_trades_5m.csv")
# files.download("yf_lab_model_trades_1m.csv")  # if fetched
```
