# Commodity momentum combined with term structure (QuantConnect)

Quantpedia #23 / Fuertes, Miffre & Rallis (2010). Runs on QuantConnect LEAN:
upload `main.py` and `selection.py` as one project. This repo has no futures
curve data, so it can't be backtested locally. Only the selection logic in
`selection.py` is unit-tested (`tests/test_commodity_momentum_term_structure.py`).

## Rules (monthly, first trading day + 30 min)

1. For each of 22 commodity futures, take the **nearest** contract and the next
   **deferred** contract with open interest that expires ≥ 35 days out.
2. Roll yield = `ln(near / deferred) * 365 / days_between_expiries`
   (positive = backwardation).
3. Sort commodities into roll-yield tertiles.
4. Within the top tertile, go **long** the half with the highest momentum. Within
   the bottom tertile, go **short** the half with the lowest momentum. Momentum is
   the 21-day mean daily return of the deferred contract.
5. Equal weight within each leg, +5% gross long and -5% gross short. Trade the
   deferred contract and liquidate everything else.

## Fixes vs. the QuantConnect library version

- Roll return was `(near - far) * 365 / days` in raw price units, so cocoa and
  gold always dominated the ranking. It's now a log ratio, so the ranking doesn't
  depend on price level.
- Every deferred contract was ranked separately, so one commodity could take
  several slots or appear on both legs. Now each commodity gets one entry.
- `floor(n/3) == 0` made `sorted[-0:]` short the whole universe, and a tertile
  of 1 gave `int(0.5) == 0`, which raised a `ZeroDivisionError`. Each leg now
  keeps at least one name, and the strategy goes flat with fewer than 3 commodities.
- The momentum indicator wasn't checked for readiness, so new contracts ranked
  on 0. Contracts without a ready indicator are now skipped.
- The traded contract has to outlive the next rebalance (the chain filter is
  widened to 180 days so bi-monthly and quarterly cycles still have a candidate).
