# TradingView -> webhook -> Tradovate, for MES

This wires the Failed-2s strategy up as: **TradingView generates the
signal** (it has live market data and runs the Pine Script), **fires a
webhook** with the signal details, and a small server you host **computes
position size from your risk budget and places the order on Tradovate**.

```
TradingView chart (Pine script)
    -> alert() fires on signal, JSON payload
    -> TradingView's alert servers POST it to your webhook URL
    -> webhook/server.py: checks secret, sizes contracts from risk budget,
       checks risk manager + open-position safety, places bracket order
    -> Tradovate (demo or live account)
```

## 1. Install the Pine script

1. On TradingView, open **Pine Editor** (bottom panel of any chart).
2. Paste in the contents of `tradingview/failed2s_mes.pine`.
3. Click **Add to Chart**.
4. Set the **chart's own timeframe** to your entry timeframe, and set the
   script's **Bias Timeframe** input to match one of the three pairs tested
   in the Python backtester:

   | chart timeframe | Bias Timeframe input |
   |---|---|
   | 1 minute | `15` |
   | 5 minutes | `60` |
   | 15 minutes | `240` |

5. Set **Tick size** to `0.25` (MES) -- already the default.
6. Set **Webhook secret** to a random string you'll also put in the
   webhook server's `WEBHOOK_SECRET` env var. This is the only thing
   stopping a random internet request from placing an order, so make it
   an actual secret, not `changeme`.
7. Sanity-check it on TradingView's own **Strategy Tester** tab before
   moving on -- this is a Pine port of the Python strategy, not guaranteed
   bar-for-bar identical (see the caveat at the top of the `.pine` file).

## 2. Create the TradingView alert

1. Right-click the chart -> **Add alert** (or the Alert clock icon).
2. **Condition**: pick the strategy name, then choose **"Any alert()
   function call"**. (Not "Order fills" -- the script calls `alert()`
   directly with the JSON payload.)
3. **Webhook URL**: your deployed server's URL, e.g.
   `https://your-host.example.com/tradingview-webhook`. Check the
   "Webhook URL" box.
4. The **Message** box content is ignored for `alert()`-based alerts --
   the JSON string built in the Pine script is what actually gets sent.
   Leave it as-is.
5. Set **Expiration** to "Open-ended" if your plan allows it, and confirm.

Webhook alerts require a TradingView plan that supports them -- check your
plan's alert limits/webhook support before relying on this.

## 3. Deploy the webhook server

The server (`webhook/server.py`) needs a public HTTPS endpoint TradingView
can reach, and outbound access to Tradovate's API -- neither of which this
sandboxed environment has, so this step happens on your own infrastructure.

**Quick test (temporary, local machine + tunnel):**
```bash
pip install -r requirements.txt -r webhook/requirements.txt
export WEBHOOK_SECRET=<same secret as the Pine script>
export TRADOVATE_ACCOUNT_SPEC=<your Tradovate username>
export TRADOVATE_USERNAME=... TRADOVATE_PASSWORD=... TRADOVATE_APP_ID=...
export TRADOVATE_CID=... TRADOVATE_SEC=...
export TRADOVATE_ENV=demo
export DRY_RUN=true          # log-only until you've verified behavior
export RISK_PER_TRADE_USD=200
export MAX_CONTRACTS=5

uvicorn webhook.server:app --host 0.0.0.0 --port 8000
# in another terminal:
ngrok http 8000              # gives you a public https:// URL for the TradingView alert
```

**Persistent (small VPS):** run the same `uvicorn` command under a
systemd service or inside Docker, behind a reverse proxy (Caddy/nginx) for
TLS, with the same environment variables set as real secrets (not
committed anywhere).

## 4. Test before it's real

1. `TRADOVATE_ENV=demo`, `DRY_RUN=true` first. Trigger a manual TradingView
   alert test (the "Test" button in the alert dialog sends a real webhook
   call) and confirm the server logs show the expected sizing math.
2. Flip to `DRY_RUN=false` with `TRADOVATE_ENV=demo` still set, and confirm
   real bracket orders appear correctly (right qty, right stop/target) on
   your **Tradovate demo account**.
3. Only after that, switch to `TRADOVATE_ENV=live`.

## Risk-based position sizing

`RISK_PER_TRADE_USD` is the dollar amount you're willing to risk per trade.
`webhook/sizing.py` computes `contracts = risk_usd / (stop_distance_points *
point_value)`, floored to a whole contract, clamped to `[1, MAX_CONTRACTS]`.
Since the stop distance varies signal to signal (it's derived from swing
structure, not fixed), contract count varies with it -- a tighter stop
gets more contracts for the same dollar risk, a wider stop gets fewer.
`MAX_CONTRACTS` is a hard ceiling regardless of that math -- set it to
whatever your prop firm's max position size actually allows.

## What this does and doesn't protect you from

- **Duplicate/stacked entries**: before placing an order, the server checks
  Tradovate's actual open position size for the account and refuses a new
  entry if it isn't flat (see the caveat on `get_position_size()` in
  `live/tradovate_client.py` if the same account trades more than MES).
- **Daily trade count**: `MAX_DAILY_TRADES` correctly rate-limits entries.
- **Daily loss limit**: `DAILY_LOSS_LIMIT_USD` does **not** currently work
  as a real backstop -- see the "Known gap" note at the top of
  `webhook/server.py`. There's no fill/close feed wired in yet, so it
  never sees real P&L. Don't treat it as your actual daily-loss protection
  until that's built; use Tradovate's/your prop firm's own account-level
  loss limit for that in the meantime.
