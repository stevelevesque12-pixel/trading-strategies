"""Offline tests for the Polymarket weather strategy (no network)."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from polyweather.calibration import Calibrator, fit
from polyweather.fees import fee_per_share, taker_fee
from polyweather.forecast import DayForecast, _hour_grid, nbm_day_forecast, parse_open_meteo_ensemble
from polyweather.markets import Bucket, WeatherEvent, event_slug, parse_bucket_label, parse_event
from polyweather.model import SourceCalibration, bucket_probabilities, default_calibration
from polyweather.observations import Obs, ObsState, day_window_utc, round_half_up
from polyweather.orderbook import OrderBook
from polyweather.risk import RiskConfig, RiskManager
from polyweather.stations import STATIONS, station_for
from polyweather.strategy import StrategyConfig, find_arbitrage, plan_event

NYC = STATIONS["nyc"]
D = date(2026, 10, 19)


def make_event(kind="high", unit="F", labels=None, precision=1.0, fee=0.05):
    labels = labels or ["51°F or below", "52-53°F", "54-55°F", "56-57°F", "58-59°F", "60-61°F",
                        "62-63°F", "64-65°F", "66-67°F", "68-69°F", "70°F or higher"]
    buckets = []
    for i, lab in enumerate(labels):
        lo, hi, _ = parse_bucket_label(lab)
        buckets.append(Bucket(lab, lo, hi, yes_token=f"y{i}", no_token=f"n{i}"))
    return WeatherEvent(slug="test", title="t", city="nyc", kind=kind, target_date=D, unit=unit,
                        station=NYC, station_code="KLGA", buckets=buckets, fee_rate=fee, precision=precision)


def flat_forecast(value, n=31, spread=0.0, source="test", shape=None):
    grid = _hour_grid(D, NYC.tz)
    base = np.full(len(grid), value - 8.0)
    base[14:17] = value  # afternoon peak (local hours 14-16)
    if shape is not None:
        base = shape
    z = np.linspace(-1.5, 1.5, n)
    return DayForecast(source, grid, base[None, :] + spread * z[:, None], datetime(2026, 10, 18, tzinfo=timezone.utc), D)


# ---------------------------------------------------------------- markets


@pytest.mark.parametrize("label,expected", [
    ("51°F or below", (None, 51.0, "F")),
    ("52-53°F", (52.0, 53.0, "F")),
    ("70°F or higher", (70.0, None, "F")),
    ("9°C", (9.0, 9.0, "C")),
    ("-3°C", (-3.0, -3.0, "C")),
    ("-5°C or below", (None, -5.0, "C")),
    ("12°C or higher", (12.0, None, "C")),
])
def test_parse_bucket_label(label, expected):
    assert parse_bucket_label(label) == expected


def test_bucket_contains_whole_and_tenths():
    b = Bucket("62-63°F", 62, 63, "y", "n")
    assert b.contains(62) and b.contains(63) and not b.contains(64) and not b.contains(61)
    hk = Bucket("31°C", 31, 31, "y", "n")
    assert hk.contains(31.0, 0.1) and hk.contains(31.9, 0.1) and not hk.contains(32.0, 0.1)


def test_parse_event_from_gamma_json():
    raw = {
        "slug": "highest-temperature-in-nyc-on-october-19-2026", "title": "Highest temperature in NYC on October 19?",
        "eventDate": "2026-10-19", "endDate": "2026-10-19T12:00:00Z", "negRisk": True, "closed": False,
        "description": "recorded by NOAA at the LaGuardia Airport Station in degrees Fahrenheit ... "
                       "https://www.weather.gov/wrh/timeseries?site=klga ... using the \"Show Hourly Data\" button.",
        "markets": [
            {"groupItemTitle": "60-61°F", "clobTokenIds": '["1","2"]', "outcomePrices": '["0.3","0.7"]',
             "feesEnabled": True, "feeSchedule": {"rate": 0.05, "exponent": 1}, "orderMinSize": 5,
             "orderPriceMinTickSize": 0.001, "acceptingOrders": True},
            {"groupItemTitle": "59°F or below", "clobTokenIds": '["3","4"]', "outcomePrices": '["0.1","0.9"]'},
        ],
    }
    ev = parse_event(raw)
    assert ev.kind == "high" and ev.city == "nyc" and ev.target_date == D
    assert ev.station.icao == "KLGA" and ev.hourly_only and ev.fee_rate == 0.05
    assert [b.label for b in ev.buckets] == ["59°F or below", "60-61°F"]  # sorted
    assert ev.tradeable


def test_station_change_in_rules_is_respected():
    # If the rules name a different station than our table, use the rules' station (or skip).
    assert station_for("nyc", "KLGA").icao == "KLGA"
    assert station_for("nyc", "KJFK") is None
    assert station_for("nyc", "KORD").city == "chicago"


def test_event_slug():
    assert event_slug("low", "los-angeles", date(2026, 9, 1)) == "lowest-temperature-in-los-angeles-on-september-1-2026"


# ---------------------------------------------------------------- fees & books


def test_fee_matches_docs_example():
    # docs.polymarket.com: 100 shares at $0.50, weather rate 0.05 -> $1.25
    assert taker_fee(100, 0.5, 0.05) == pytest.approx(1.25)
    assert fee_per_share(0.1, 0.05) == pytest.approx(0.0045)
    assert taker_fee(100, 0.5, 0.0) == 0.0


def test_sweep_cost_walks_levels():
    ob = OrderBook("t", bids=[], asks=[(0.40, 10), (0.45, 10)])
    filled, cost, worst = ob.sweep_cost(15)
    assert filled == 15 and worst == 0.45 and cost == pytest.approx(10 * 0.40 + 5 * 0.45)


# ---------------------------------------------------------------- observations


def test_round_half_up_and_obs_state():
    assert round_half_up(55.5) == 56 and round_half_up(55.49) == 55 and round_half_up(-0.5) == 0
    start, end = day_window_utc(D, NYC.tz)
    times = [start + timedelta(hours=h) for h in range(10)]
    obs = Obs(times, [50, 51, 53, 52, 55, 54, 54, 56, 55, 57])
    st = ObsState.from_obs(obs, "high", times[7])
    assert st.extreme == 56 and st.last_value == 56 and st.n == 8
    assert ObsState.from_obs(obs, "low", times[7]).extreme == 50


def test_day_window_handles_dst():
    s, e = day_window_utc(date(2026, 11, 1), "America/New_York")  # fall back: 25-hour day
    assert (e - s) == timedelta(hours=25)


# ---------------------------------------------------------------- model


def test_probabilities_sum_to_one_and_peak_at_forecast():
    ev = make_event()
    fc = flat_forecast(62.4, spread=1.0)
    cal = SourceCalibration(kernel_sd=1.5)
    out = bucket_probabilities(ev, [(fc, cal)], None, fc.times[0] - 86400, NYC.tz)
    assert out.probs.sum() == pytest.approx(1.0)
    assert int(np.argmax(out.probs)) == 6  # 62-63
    assert out.expected == pytest.approx(62.4, abs=0.6)


def test_observed_high_floors_the_distribution():
    ev = make_event()
    fc = flat_forecast(60.0, spread=0.5)
    start, _ = day_window_utc(D, NYC.tz)
    asof = start + timedelta(hours=17)  # after the forecast peak
    obs = ObsState(extreme=64, last_value=63, last_time=asof - timedelta(minutes=9), n=17, asof=asof)
    out = bucket_probabilities(ev, [(fc, SourceCalibration(kernel_sd=1.5))], obs, asof.timestamp(), NYC.tz)
    assert out.probs[:7].sum() == pytest.approx(0.0, abs=1e-9)  # everything below 64 is dead
    assert out.probs[7] > 0.9  # 64-65


def test_day_over_is_point_mass():
    ev = make_event()
    fc = flat_forecast(60.0)
    _, end = day_window_utc(D, NYC.tz)
    obs = ObsState(extreme=61, last_value=55, last_time=end - timedelta(minutes=9), n=24, asof=end)
    out = bucket_probabilities(ev, [(fc, SourceCalibration())], obs, end.timestamp() + 60, NYC.tz)
    assert out.final and out.probs[5] == 1.0


def test_low_market_mirrors():
    labels = ["40°F or below", "41-42°F", "43-44°F", "45-46°F", "47°F or higher"]
    ev = make_event(kind="low", labels=labels)
    grid = _hour_grid(D, NYC.tz)
    shape = np.full(len(grid), 55.0)
    shape[5:7] = 43.4
    fc = flat_forecast(0, spread=0.3, shape=shape)
    out = bucket_probabilities(ev, [(fc, SourceCalibration(kernel_sd=1.0))], None, grid[0] - 3600, NYC.tz)
    assert int(np.argmax(out.probs)) == 2
    # observed low of 41 so far: buckets above 41-42 are dead
    obs = ObsState(extreme=41, last_value=45, last_time=datetime.fromtimestamp(grid[8], timezone.utc),
                   n=9, asof=datetime.fromtimestamp(grid[8], timezone.utc))
    out = bucket_probabilities(ev, [(fc, SourceCalibration(kernel_sd=1.0))], obs, grid[8], NYC.tz)
    assert out.probs[2:].sum() == pytest.approx(0.0, abs=1e-9)


def test_tenths_precision_buckets():
    ev = make_event(unit="C", labels=["29°C or below", "30°C", "31°C", "32°C", "33°C or higher"], precision=0.1)
    fc = flat_forecast(31.5, spread=0.0)
    out = bucket_probabilities(ev, [(fc, SourceCalibration(kernel_sd=0.2))], None, fc.times[0] - 86400, NYC.tz)
    assert out.probs[2] > 0.95  # 31.5 sits in "31°C" = [31.0, 32.0)


def test_mixture_weights():
    ev = make_event()
    a = flat_forecast(56.5, source="a")  # centre of 56-57
    b = flat_forecast(64.5, source="b")  # centre of 64-65
    t0 = a.times[0] - 86400
    out = bucket_probabilities(ev, [(a, SourceCalibration(kernel_sd=0.5, weight=3.0)),
                                    (b, SourceCalibration(kernel_sd=0.5, weight=1.0))], None, t0, NYC.tz)
    # each source's mass mostly lands in its own bucket; the split follows the 3:1 weights
    assert out.probs[3] / out.probs[7] == pytest.approx(3.0, rel=0.01)
    assert out.probs[3] + out.probs[7] > 0.9


# ---------------------------------------------------------------- forecast parsing


def test_parse_open_meteo_ensemble():
    payload = {"hourly": {
        "time": ["2026-10-19T00:00", "2026-10-19T01:00"],
        "temperature_2m_ecmwf_ifs025_ensemble": [10.0, 11.0],
        "temperature_2m_member01_ecmwf_ifs025_ensemble": [10.5, 11.5],
        "temperature_2m_ncep_gefs025": [9.0, None],
    }}
    out = parse_open_meteo_ensemble(payload)
    t, m = out["ecmwf_ens"]
    assert m.shape == (2, 2) and m[1, 1] == 11.5
    assert np.isnan(out["gefs"][1][0, 1])


def test_nbm_paths_center_on_txn():
    start, _ = day_window_utc(D, NYC.tz)
    rows = []
    for h in range(0, 54, 3):
        t = start - timedelta(hours=start.hour) + timedelta(hours=h)  # 00Z-aligned 3-hourly
        rows.append({"ftime_utc": t.strftime("%Y-%m-%dT%H:%M:%S"), "tmp": 55 + 6 * np.sin((h - 9) / 24 * 2 * np.pi),
                     "tsd": 2.0, "txn": None, "xnd": None})
    for r in rows:
        if r["ftime_utc"].startswith("2026-10-20T00:00"):
            r["txn"], r["xnd"] = 66.0, 2.5
    f = nbm_day_forecast(rows, datetime(2026, 10, 18, 12, tzinfo=timezone.utc), NYC, D, "high")
    daily_max = np.nanmax(f.paths, axis=1)
    assert np.median(daily_max) == pytest.approx(66.0, abs=0.3)
    assert np.std(daily_max) == pytest.approx(2.5, rel=0.25)


# ---------------------------------------------------------------- strategy


def _books(ev, yes_prices, depth=1000.0, half=0.01):
    books = {}
    for b, p in zip(ev.buckets, yes_prices):
        books[b.yes_token] = OrderBook(b.yes_token, [(max(p - half, 0.001), depth)], [(min(p + half, 0.999), depth)])
        books[b.no_token] = OrderBook(b.no_token, [(max(1 - p - half, 0.001), depth)], [(min(1 - p + half, 0.999), depth)])
    return books


def test_no_trade_when_market_agrees():
    ev = make_event()
    p = np.array([0.01, 0.02, 0.05, 0.1, 0.17, 0.25, 0.2, 0.12, 0.05, 0.02, 0.01])
    plan = plan_event(ev, p, _books(ev, p), 1000, [], StrategyConfig())
    assert plan.orders == []


def test_buys_underpriced_bucket_and_respects_caps():
    ev = make_event()
    model = np.array([0.0, 0.0, 0.01, 0.04, 0.15, 0.45, 0.25, 0.08, 0.02, 0.0, 0.0])
    market = np.array([0.0, 0.0, 0.02, 0.06, 0.2, 0.2, 0.3, 0.15, 0.05, 0.01, 0.01])
    cfg = StrategyConfig(model_weight=1.0)
    plan = plan_event(ev, model, _books(ev, market), 1000, [], cfg)
    buys = [o for o in plan.orders if o.action == "BUY"]
    assert any(o.bucket == 5 and o.side == "YES" for o in buys)
    assert sum(o.est_cost for o in buys) <= cfg.max_event_frac * 1000 + 1e-6
    for o in buys:
        assert o.edge >= cfg.min_edge
    # fee is included in the cost
    o = next(o for o in buys if o.bucket == 5 and o.side == "YES")
    assert o.est_cost == pytest.approx(o.shares * (0.21 + fee_per_share(0.21, 0.05)), rel=1e-3)


def test_fee_can_kill_a_thin_edge():
    ev = make_event()
    model = np.full(11, 0.0)
    model[5], model[6] = 0.555, 0.445
    market = np.full(11, 0.0)
    market[5], market[6] = 0.50, 0.50
    cfg = StrategyConfig(model_weight=1.0, min_edge=0.03, min_edge_frac=0.0, prob_floor=0.0)
    # all-in YES ask = 0.51 + 0.05*0.51*0.49 = 0.5225 -> edge 0.0325 >= 0.03 with no fee floor...
    assert plan_event(ev, model, _books(ev, market), 1000, [], cfg).orders
    # ...but with min_edge 0.04 the fee makes it untradeable
    cfg.min_edge = 0.04
    assert not plan_event(ev, model, _books(ev, market), 1000, [], cfg).orders


def test_dead_buckets_from_obs_are_sold_via_no():
    ev = make_event()
    model = np.zeros(11)
    model[7] = 1.0  # obs already at 64, model certain
    market = np.array([0, 0, 0, 0, 0, 0.0, 0.3, 0.6, 0.08, 0.01, 0.01])
    cfg = StrategyConfig(model_weight=1.0)
    plan = plan_event(ev, model, _books(ev, market), 1000, [], cfg, observed_extreme=64)
    # Market still prices 62-63 at 0.30 though obs have ruled it out: back the live bucket.
    assert any(o.bucket == 7 and o.side == "YES" for o in plan.orders)
    assert plan.q[6] == pytest.approx(cfg.dead_adjacent_prob, rel=0.2)
    assert plan.q[:6].max() <= cfg.dead_other_prob * 1.1
    # Without the YES leg available, the same view is expressed as NO on the dead bucket.
    books = _books(ev, market)
    books["y7"].asks = []
    plan = plan_event(ev, model, books, 1000, [], cfg, observed_extreme=64)
    assert any(o.bucket == 6 and o.side == "NO" for o in plan.orders)


def test_existing_position_reduces_new_size():
    from polyweather.ledger import Position
    ev = make_event()
    model = np.array([0.0, 0.0, 0.01, 0.04, 0.15, 0.45, 0.25, 0.08, 0.02, 0.0, 0.0])
    market = np.array([0.0, 0.0, 0.02, 0.06, 0.2, 0.2, 0.3, 0.15, 0.05, 0.01, 0.01])
    cfg = StrategyConfig(model_weight=1.0, max_event_frac=1.0)
    fresh = plan_event(ev, model, _books(ev, market), 1000, [], cfg)
    held = [Position("test", 5, "YES", "y5", "nyc", "high", str(D), "60-61°F", 300.0, 66.0, False, 0.0)]
    again = plan_event(ev, model, _books(ev, market), 1000, held, cfg)
    yes5 = lambda plan: sum(o.shares for o in plan.orders if o.bucket == 5 and o.side == "YES" and o.action == "BUY")
    assert yes5(again) < yes5(fresh)


def test_neg_risk_arbitrage():
    ev = make_event(labels=["59°F or below", "60-61°F", "62°F or higher"], fee=0.0)
    books = {}
    for b, a in zip(ev.buckets, [0.30, 0.30, 0.30]):
        books[b.yes_token] = OrderBook(b.yes_token, [], [(a, 100)])
        books[b.no_token] = OrderBook(b.no_token, [], [(0.75, 100)])
    orders = find_arbitrage(ev, books, StrategyConfig())
    assert len(orders) == 3 and all(o.side == "YES" and o.reason == "arb" for o in orders)


def test_risk_budget_city_day_cap():
    from polyweather.ledger import Position
    rm = RiskManager(RiskConfig(max_city_day_frac=0.06), bankroll=1000)
    ev = make_event()
    held = [Position("other", 1, "YES", "x", "nyc", "low", str(D), "l", 100, 50.0, False, 0.0)]
    assert rm.budget_for(ev, held) == pytest.approx(10.0)
    rm.day_pnl = -200
    assert rm.budget_for(ev, held) == 0.0


# ---------------------------------------------------------------- calibration


def test_fit_recovers_bias_and_spread():
    rng = np.random.default_rng(0)
    centers = rng.normal(60, 8, 400)
    obs = np.round(centers - 1.0 + rng.normal(0, 2.0, 400))
    prior = default_calibration("nbm", 1, "F", "high", False)
    cal = fit([(c, 0.0, o) for c, o in zip(centers, obs)], prior)
    assert cal.bias == pytest.approx(-1.0, abs=0.25)
    assert cal.kernel_sd == pytest.approx(2.0, abs=0.3)
    assert cal.n > 350


def test_calibrator_fallbacks(tmp_path):
    cal = Calibrator(str(tmp_path / "c.json"))
    cal.set("nyc", "high", "nbm", 1, SourceCalibration(bias=-1.2, kernel_sd=0.5, weight=5.0, n=40))
    cal.save()
    cal2 = Calibrator(str(tmp_path / "c.json"))
    assert cal2.get("nyc", "high", "nbm", 1, "F", True).bias == -1.2
    assert cal2.get("nyc", "high", "nbm", 0, "F", True).bias == -1.2  # lead 0 borrows lead 1's bias
    assert cal2.get("nyc", "low", "nbm", 1, "F", True).n == 0  # default
