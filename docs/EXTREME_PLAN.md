# Extreme plan

Trade the MNQ 5-min ORB (PropQuantX trade list) on Tradeify Growth 50K
accounts across up to 5 prop firms, 5 funded accounts per firm (25 total).
Simulator: `python -m prop_sim.extreme`.

## Rules

1. **Size 10 / 10 / 6 MNQ.** Eval at 10 micros, funded at 10 until the first
   payout, then 6.
2. **Fill every empty slot with an eval.** A firm with 2 funded accounts runs
   3 evals at once. A breached eval or funded account is replaced the same day
   with a new eval at the same firm.
3. **Open firms in turn.** When a firm holds 5 funded accounts, start evals at
   the next firm. Firms already open stay open.
4. **Request every payout as soon as it is allowed** (balance at $53,000+,
   5 days of $150+, best day <= 35% of profit since the last payout).
5. **Cap eval spending.** Suggested: pause new evals after about $1,500 a month
   without a payout, so a 2020-21-type year cannot run up a $25,000 fee bill.
   (Not simulated.)

## Simulated results

MNQ ORB trade list May 2019 - Sep 2026, $1.50 per micro round trip, $145 per
eval, every firm modelled as Tradeify Growth 50K. One-year windows of real days
in order. All 25 accounts trade the same signal.

| | 2023-2026 markets | 2019-2026 markets |
| --- | --- | --- |
| Payouts per year (mean / median) | 86 / 80 | 39 / 10 |
| Net per year, mean | $120,903 | $44,740 |
| Net per year, median | $104,534 | -$6,525 |
| Worst tenth of years | -$3,990 | -$18,850 |
| Best tenth of years | +$306,441 | +$220,671 |
| Worst year | -$23,200 | -$25,375 |
| Losing years | 16% | 59% |
| Evals per year (fees) | ~106 (~$15,300) | ~111 (~$16,000) |
| Funded accounts held, average | 10.0 | 5.6 |

Run continuously from January 2023 to September 2026: 415 payouts, $599,228
net after 455 evals; all 25 accounts funded after about 335 trading days
(~16 months); 12.6 funded accounts held on average.

Variants (`--one-eval`, `--all-open`, `--plan`):

| Variant, 2023-2026 | Net per year (mean / median) | Losing years |
| --- | --- | --- |
| Extreme (this plan) | $120,903 / $104,534 | 16% |
| One eval at a time per firm | $22,081 / $11,987 | 5% |
| All 5 firms open on day one | $178,997 / $175,165 | 2% |
| Funded 5 / 5 instead of 10 / 6 | $76,331 / $69,708 | 27% |

## Risks

- **25 accounts are one position.** They pass, pay and fail on the same days.
- **Weak markets are expensive.** From 2019, 59% of years lose money, mostly
  eval fees with few payouts; the worst year is about -$25,000.
- **2025 carries most of the 2023-2026 profit.**
- **Rules are assumed.** Withdrawal amount (balance above $51,500, up to the
  cap), cap steps ($1,500 / $2,000 / $2,500 / $3,000), drawdown lock at
  $50,100, and the daily loss limit closing trades at -$1,250 are not confirmed
  with Tradeify. Other firms are treated as copies of Tradeify Growth; their
  real rules, prices, account limits and copy-trading policies differ.
- Simulated results from one trade list, not guaranteed.
