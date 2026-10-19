# polyweather: Polymarket daily-temperature trading

An end-to-end system for Polymarket's "Highest / Lowest temperature in
&lt;city&gt; on &lt;date&gt;?" markets: market discovery, station-exact forecasts,
observation-conditioned bucket probabilities, Kelly sizing against real
order books with fees netted, portfolio risk limits, paper/live execution,
a ledger that doubles as a calibration dataset, and a walk-forward
backtester on resolved markets.

**Bottom line first:** the system works mechanically, and its settlement
model is validated (99% of resolutions reproduced). But no version of the
trading strategy showed a reliable edge once the backtest was made
honest (fair information timing, fills only against trades that really
printed). Treat it as research infrastructure and paper-trade it. The
results are in [What the backtests say](#what-the-backtests-say).

## The markets

* ~50 cities, one "highest" and one "lowest" event per city per day, listed
  ~2 days ahead. Each event has ~11 mutually exclusive buckets (`52-53°F`,
  `9°C`, `70°F or higher`, ...), and each bucket is a neg-risk YES/NO pair on
  the CLOB.
* **Settlement** is the max/min of the named airport station's METAR
  readings over the *local calendar day*, as shown on
  `weather.gov/wrh/timeseries?site=<icao>`: whole °F in the US, whole °C
  elsewhere. Some events (e.g. NYC since autumn 2026) count routine hourly
  readings only ("Show Hourly Data"), and that's parsed per event. Hong
  Kong settles on the HKO daily extract instead and is excluded from
  trading.
* **Fees:** taker-only, `fee = shares x 0.05 x p x (1 - p)`. That's 1.25c/share
  at p=0.5, but 4.5% of notional at p=0.10. Makers pay nothing.

## How it works

```
markets.py      Gamma API -> WeatherEvent (buckets, tokens, station, fee schedule, rule flags)
stations.py     50 settlement stations (ICAO, coords, elevation, tz, unit)
forecast.py     DayForecast = M member hourly paths over the target local day
                  * NOAA NBM (US) via IEM: TXN/XND + TMP/TSD -> quantile pseudo-members,
                    newest run backfilled from the previous one
                  * Open-Meteo ensembles (global): ECMWF 51, GEFS 31, ICON 40, GEM 21, UKMO 18
observations.py METARs (IEM archive / aviationweather live) for the settled extreme so far,
                plus the 5-minute ASOS feed (IEM HF archive / api.weather.gov live) as the
                nowcast anchor
model.py        per member: extreme over the *remaining* hours, nudged to the latest reading,
                bias-corrected, kernel-dressed; V = max(observed_so_far, round(X)) gives a
                closed-form distribution over the resolved grid -> bucket probabilities;
                sources mixed by weight
calibration.py  per station/kind/source/lead: bias, spread scale, kernel, weight (shrunk);
                same-day: (bias, sd) of the remaining-day extreme by local hour
strategy.py     q = w*model + (1-w)*market (per bucket, only where the book is informative;
                w drops for uncalibrated sources); event-level fractional-Kelly: greedy
                max E[log W] over all YES and NO legs, walking real ask depth incl. fees;
                exits when the bid overshoots fair; neg-risk set arbitrage
risk.py         caps per event / city-day / region-day / total, daily loss stop
execution.py    PaperBroker (default) | LiveBroker (py-clob-client, FAK limit orders)
ledger.py       SQLite: forecasts, model vs book at every look, orders, positions, resolutions
runner.py       discover -> evaluate -> plan -> risk -> execute -> log; settle loop
backtest.py     walk-forward backtest, resolution validation, NBM archive calibration
```

Details that matter (each was found by checking against data, not assumed):

* **The station, not the city.** LaGuardia, London City and Le Bourget are
  what settle. Forecasts are fetched for those points at station elevation.
  If a market's rules name a different station than the table, the rules
  win, or the event is skipped.
* **Full local calendar day.** Highs sometimes come at 00:51 (frontal days),
  and lows often at 23:59. That's why forecasts are whole hourly paths, not
  "daytime max".
* **Hourly sampling bias.** Routine-only settlement under-reads the true
  max. Calibrated NBM highs settle 0.2-1.5°F below forecast at 10 of 11 US
  stations (LAX is the exception).
* **NBM's 6-hour blind spot.** A bulletin's first projection is 6h after its
  cycle time, so the newest run alone can't see the afternoon you're
  trading. Hours are backfilled from the previous run.
* **Intraday uncertainty.** Anchoring members to the current reading
  collapses ensemble spread, so the remaining error is carried by an
  empirical (bias, sd) per local hour. Even at 16:00 the remaining-day
  high is uncertain by ±1.7°F.

## Validation: can we reproduce the settlement?

`python -m polyweather validate --days N` rebuilds each resolved event's
value from archived METARs and checks the winning bucket:

* all 50 METAR cities, 6 days: **548/552 = 99.3%**
* 11 US cities, 120 days: **2049/2072 = 98.9%**

The misses are single readings that differ from the official page. The
strategy keeps 1% / 0.1% probability on observation-"dead" buckets for
this reason, rather than treating a 0.95 NO as free money.

## What the backtests say

11 US cities (the ones with an archived NBM), Jun 19 – Oct 16 2026, about
2,070 resolved events. Every input is what was public at the decision time:
the NBM bulletin, the METARs, the market price, and a calibration refit
only on earlier resolutions. Global ensembles are left out because their
history isn't archived. The 120 days are split into a training half (Jun
19 – Aug 17) used for design choices and a held-out test half (Aug 18 –
Oct 16).

The headline result is how much it depends on backtest honesty:

| Backtest assumptions | ROI on staked |
|---|---|
| Fill at mid ±1c, $150 depth, obs vs price timing loose | +16.0% |
| ... spreads/depth measured from live books | +5.8% |
| ... 75th-percentile measured spreads | +0.5% |
| Strict timing (model sees only METARs ≥5 min older than the price) + intraday fixes, measured spreads: train / test | −2.3% / +6.8% |
| **Fills only against real taker prints (50% capture): train / test** | **+3.3% / −15.8%** |

What each step taught:

1. **"Lowest temperature" books are nearly empty.** In a live snapshot of all
   247 open events, lows had median spreads of 30-80c near the money with
   $10-25 of depth. A historical mid of a 0.30/0.70 book isn't a price. All
   of the apparent low-market edge (+20-45% in looser backtests) vanished
   once fills had to match real trades: about $200-350 total could have
   filled in 60 days.
2. **The market sees the 5-minute feed.** The weather.gov page that
   settles these markets shows 5-minute ASOS data, and traders watch it. A
   model that compares the latest hourly METAR to a slightly older price
   looks brilliant (+24-30%). With timing made fair it loses. Adding the
   5-minute feed to the model didn't close the gap.
3. **Day-ahead, NBM alone doesn't beat the price.** Market Brier score is
   0.67-0.74 vs the model's 0.74-0.75 at 1-day lead. PnL is −7%/−13% (train)
   and ≈0% (test) with measured-spread fills, and +5%/−4% with real-print
   fills: noise around zero.
4. **Same-day, the blend forecasts better than the mid but still doesn't
   trade profitably.** On the test half the blended probability beats the
   market mid's Brier score at every same-day hour (0.32 vs 0.36 at 15:00).
   But with real-print fills, PnL is −16% on $17k staked. The prints that
   would have filled us are adversely selected. Capacity is also tiny:
   ~$300/day across 11 cities at the prices the model wants.

So the honest conclusion: **this taker strategy doesn't have a demonstrated
edge**. The forecasting and settlement machinery is sound and reusable.
The most promising directions, none of them tested yet:

* **Market making** instead of taking. Makers pay no fee and get a 25%
  rebate share. Quote around the blended fair value and pull quotes when
  the 5-minute feed moves. The trade-print data in `backtest.py` can drive
  a fill simulation for resting orders.
* **Faster/better nowcast inputs:** hourly LAMP (IEM keeps only a few days
  of hourly runs, so it can't be backtested yet), HRRR, 1-minute ASOS.
* **International cities.** Many settle on 30-minute METARs with no public
  5-minute feed, so the information race is different. The live system
  already trades them on Open-Meteo ensembles, but there's no forecast
  archive to backtest with. The ledger collects that data as it runs.
* Collect real order-book history (the ledger logs best bid/ask at every
  look) so future backtests don't need a cost model at all.

## Running it

```bash
pip install -r requirements.txt            # numpy, pandas, requests (+ pytest for tests)

python -m polyweather scan                 # model vs book for every open event, no orders
python -m polyweather scan --cities nyc chicago london
python -m polyweather trade --loop 300     # PAPER trading every 5 min -> polyweather.db
python -m polyweather settle               # settle paper positions, record outcomes
python -m polyweather calibrate --nbm-days 90       # refit US NBM calibration
python -m polyweather calibrate --from-ledger       # fit any source from the bot's own forecast log
python -m polyweather validate --days 14
python -m polyweather backtest --days 60 --end 2026-10-16 \
    --decisions 0@13 0@15 0@17 --fills trades      # LEAD@LOCALHOUR decision points
```

Useful knobs: `--model-weight` (trust in the model vs market),
`--kelly` (fraction, default 0.25), `--min-edge` ($/share after fees,
default 0.04), `--max-lead` (days ahead to trade, default 2),
`--costs measured|conservative|flat` and `--fills trades|book` for the
backtester.

Data notes:
* Open-Meteo's free tier has a daily request quota per IP. In shared cloud
  environments it may already be spent; set `OPEN_METEO_API_KEY` to use the
  customer endpoints. NBM and observations come from IEM and NOAA, with
  no key needed.
* Responses that can't change (archived forecasts, past obs, resolved
  markets) are cached under `~/.cache/polyweather` (`POLYWEATHER_CACHE`).
* `calibration.json` ships fitted NBM parameters for the 11 US stations
  (90 days to 2026-10-17).

## Live trading

Not recommended given the results above. If you do:

```bash
pip install py-clob-client
export POLY_PRIVATE_KEY=...        # signing key
export POLY_FUNDER=0x...           # your Polymarket proxy wallet (holds the USDC)
export POLY_SIGNATURE_TYPE=1       # 0 EOA, 1 email/Magic login, 2 browser-wallet proxy
export POLYWEATHER_LIVE=1          # second, explicit switch
python -m polyweather trade --live --bankroll 200 --loop 300
```

Orders are FAK limit orders at the worst level the planner priced, so
nothing rests on the book and nothing fills above the computed edge.
Polymarket's international exchange marks these markets `restricted` and
geoblocks some jurisdictions, including the US. Check that you're
eligible before connecting a wallet.
