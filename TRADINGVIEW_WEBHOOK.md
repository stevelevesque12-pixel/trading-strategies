# TradingView -> TradersPost -> Tradovate, for MES

This wires the Failed-2s strategy up as: **TradingView generates the
signal** (it has live market data and runs the Pine Script, which also
computes position size from a risk budget), **fires a webhook**, and
**TradersPost** (a hosted bridge service) executes the order on Tradovate.

```
TradingView chart (Pine script)
    -> alert() fires on signal: ticker, action, quantity, stopLoss, takeProfit
    -> TradingView's alert servers POST it to your TradersPost webhook URL
    -> TradersPost: authenticates via the URL itself, places the order
    -> Tradovate (connected with your regular login -- see why below)
```

**Why TradersPost and not a direct Tradovate API connection:** Tradovate's
direct API requires a paid "API Access Add-On" that needs a live funded
account with a $1,000+ minimum -- not available on a prop-firm sim account.
TradersPost connects to Tradovate using your regular login credentials
instead, sidestepping that requirement entirely, and is specifically
documented as a supported path for Tradeify. (The self-hosted path this
repo also has -- `webhook/server.py` talking directly to Tradovate's API --
is kept for later, in case you ever do get direct API access; see the
bottom of this doc.)

**Cost**: TradersPost is a paid subscription (Starter plan around $49/mo
at time of writing -- confirm current pricing on their site, this changes).
Confirm that's worth it for your trading volume before committing.

**Tradeify's automation rules** (confirm these still match their current
policy before going live): bots/algos are allowed on both evaluation and
funded accounts, but you must be the sole owner/user of the strategy (no
sharing across firms or with others), high-frequency trading is banned
(at least 50% of trades/profit must come from positions held >10 seconds
-- this strategy's holding times are comfortably longer than that), and
they require a live video of you enabling the code on your own PC as part
of verification.

**Tradeify's hedging rule** (from their "Hedging & Correlated Products"
page, Aug 19 2026 -- applies to Evaluation, Sim Funded and Elite Live):
you may not hold opposing positions (one long, one short) on the same
instrument **or on two products in the same Product Group**, in one account
**or across any of your accounts**. Contract size doesn't matter, so long
MES + short NQ is a hedge. Both Pine strategies trade MES or MNQ, and
those (along with ES, NQ, MYM and M2K) are all in the **Equity Index**
group. For these strategies the rule boils down to: **all your accounts must be long, short or flat on
equity index futures together -- never mixed.**

- Allowed: long the same thing on several accounts (copy trading); long MES
  + long MNQ; equity index one way + crude or gold the other way (different
  groups).
- Violation: Failed-2s long MES on account A while structure_scalp is short
  MNQ on account B; a manual NQ short alongside a bot long; trading a
  failed account against an active one.
- Consequences: every account involved goes into violation status, profits
  from the violating period are forfeited, and you can be banned
  permanently. The automatic breach check fires when a hedge lasts **over 10
  seconds AND makes over $150 profit**, but cross-account and
  correlated-product hedges are also reviewed by hand. The 10 seconds are
  for correcting a mistake, not an allowance.
- Minis and micros can now be held together, but an account's combined
  position counts as 10 micros = 1 mini against its contract limit.

**Nothing on the TradersPost path enforces this.** Each Pine script only
knows about its own chart's position. If you run
`tradingview/failed2s_mes.pine` and `tradingview/structure_scalp.pine` at
the same time, even on separate accounts, they can and eventually will
take opposite sides. Until you're on the self-hosted path (which does
check -- see the bottom of this doc), pick one of these:

1. Run **one** equity-index strategy at a time, copied to as many accounts
   as you like (same direction).
2. If you want both strategies running, put them on the **same** account
   and check TradersPost's strategy settings for an option to reject
   signals while a position is open. Confirm it works on demo first.
3. Never place manual equity-index trades while a bot can enter, and make
   sure every account is flat before you start a session.

## 1. Install/configure the Pine script

1. On TradingView, open **Pine Editor**, paste in
   `tradingview/failed2s_mes.pine`, click **Add to Chart**.
2. Set the **chart's own timeframe** to your entry timeframe, and set the
   script's **Bias Timeframe** input to match one of the three pairs tested
   in the Python backtester:

   | chart timeframe | Bias Timeframe input |
   |---|---|
   | 1 minute | `15` |
   | 5 minutes | `60` |
   | 15 minutes | `240` |

3. Under the **Position sizing** input group, set:
   - **Risk per trade (USD)** -- how much you're willing to risk per trade
   - **Point value (USD/point)** -- leave at `5` for MES
   - **Max contracts** -- a hard cap regardless of the risk math; set this
     to whatever your Tradeify account's max position size actually allows
4. Sanity-check it on TradingView's own **Strategy Tester** tab before
   moving on -- this is a Pine port of the Python strategy, not guaranteed
   bar-for-bar identical (see the caveat at the top of the `.pine` file).

## 2. Set up TradersPost

1. Sign up at `traderspost.io`.
2. Create a new **Strategy**.
3. Add a **Broker Connection** for Tradovate -- this asks for your regular
   Tradovate/Tradeify login (username + password), not API keys.
4. Once connected, TradersPost will show your available Tradovate accounts
   -- select the one tied to your Tradeify account.
5. Copy the strategy's unique **webhook URL**
   (`https://webhooks.traderspost.io/trading/webhook/<uuid>/<password>`).
   This URL *is* your authentication -- there's no separate secret field in
   the JSON payload, so keep this URL private.
6. Check your symbol mapping in the strategy settings -- confirm the
   ticker TradingView sends (e.g. `MES1!`) maps to the correct Tradovate
   contract on their end.

## 3. Create the TradingView alert

1. Right-click the chart -> **Add alert**.
2. **Condition**: pick the strategy name, then choose **"Any alert()
   function call"**. (Not "Order fills".)
3. **Webhook URL**: paste the TradersPost URL from step 2.5.
4. The **Message** box content is ignored for `alert()`-based alerts -- the
   JSON built in the Pine script is what actually gets sent. Leave it as-is.
5. Set **Expiration** to "Open-ended" if your plan allows it, confirm.

Webhook alerts require a TradingView plan that supports them -- check your
plan's alert/webhook limits.

## 4. Test before it's real

1. **First, connect TradersPost to a free Tradovate demo account** (not
   your Tradeify account) -- anyone can create one directly with Tradovate
   at no cost. Validate the whole pipeline against that first: trigger the
   alert's "Test" button, confirm TradersPost shows the signal received,
   confirm an order appears on the Tradovate demo account with the right
   quantity/stop/target.
2. Only once that's clean, switch the TradersPost broker connection to
   your actual Tradeify-provisioned Tradovate account and repeat the same
   check with real (very small) size.
3. Watch a handful of live signals closely before trusting it unattended.

## Risk-based position sizing

`riskPerTradeUsd` is the dollar amount you're willing to risk per trade.
The script computes `quantity = riskUsd / (stopDistancePoints *
pointValueUsd)`, floored to a whole contract, clamped to `[1,
maxContracts]`. Since the stop distance varies signal to signal (it's
derived from swing structure, not fixed), contract count varies with it --
a tighter stop gets more contracts for the same dollar risk, a wider stop
gets fewer.

## What this does and doesn't protect you from

- **Duplicate/stacked entries**: guarded on the TradingView side --
  `strategy.position_size == 0` gates new signals, so the script itself
  won't fire a second entry while it thinks a position is open. This is
  Pine's own (not Tradovate's real) position state, so it can drift from
  reality if an order doesn't fill the way Pine assumes.
- **Hedging across strategies/accounts**: not enforced on this path -- see
  "Tradeify's hedging rule" above.
- **Daily loss limit / daily trade cap**: not enforced anywhere in this
  path currently -- TradersPost may have its own risk controls in its
  strategy settings worth configuring; otherwise rely on Tradeify's
  account-level daily loss limit as the real backstop.

---

## Alternative: self-hosted webhook (needs direct Tradovate API access)

If you ever get direct Tradovate API credentials (CID/Secret) -- e.g. after
reaching a live/funded stage with the required minimum, or a firm that
provisions them -- this repo also has a self-hosted path that skips
TradersPost's subscription entirely: `webhook/server.py` (FastAPI) talks
to Tradovate directly via `live/tradovate_client.py`, with the same
risk-based sizing logic in `webhook/sizing.py`. It needs its own Pine
script variant (the current `.pine` file is TradersPost-formatted) and a
public host for the server (Railway config: `nixpacks.toml`).

The self-hosted path **does** enforce the hedging rule
(`failed2s/hedging.py`, used by both `webhook/server.py` and
`live/runner.py`). Before every entry it fetches open positions on every
account the Tradovate login can see and blocks the order if it would oppose
any position in the same Product Group. If the positions can't be fetched, it
blocks the entry instead of placing it unchecked. Set
`ACCOUNT_CONTRACT_LIMIT_MINIS` to your account's contract limit and it also
enforces the 10-micros-=-1-mini cap. Its blind spot is anything outside that
login: accounts behind a different login, TradersPost, or manual trades on
another platform. Keep everything you trade on one login, or keep those
other accounts flat. Ask if you
want that path built out again when you're there.
