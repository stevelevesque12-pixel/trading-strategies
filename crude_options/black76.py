"""Black-76 pricing for options on futures (CL/LO, MCL options)."""

import math

SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _d1_d2(f: float, k: float, t: float, vol: float):
    sv = vol * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * vol * vol * t) / sv
    return d1, d1 - sv


def price(f: float, k: float, t: float, vol: float, opt: str, r: float = 0.0) -> float:
    """Premium per unit (USD/bbl). `opt` is 'P' or 'C'. At/after expiry returns intrinsic."""
    if t <= 0 or vol <= 0:
        return max(k - f, 0.0) if opt == "P" else max(f - k, 0.0)
    d1, d2 = _d1_d2(f, k, t, vol)
    df = math.exp(-r * t)
    if opt == "C":
        return df * (f * _ncdf(d1) - k * _ncdf(d2))
    return df * (k * _ncdf(-d2) - f * _ncdf(-d1))


def delta(f: float, k: float, t: float, vol: float, opt: str, r: float = 0.0) -> float:
    if t <= 0 or vol <= 0:
        itm = (f > k) if opt == "C" else (f < k)
        return (1.0 if opt == "C" else -1.0) if itm else 0.0
    d1, _ = _d1_d2(f, k, t, vol)
    df = math.exp(-r * t)
    return df * _ncdf(d1) if opt == "C" else -df * _ncdf(-d1)


def strike_for_delta(f: float, t: float, vol: float, opt: str, target: float,
                     tick: float = 0.5, r: float = 0.0) -> float:
    """
    Listed strike (multiple of `tick`) whose |delta| is nearest `target` --
    the 'delta near' strike selection. Searches the strike grid around the
    analytic strike rather than inverting exactly, since only listed strikes trade.
    """
    sv = vol * math.sqrt(t)
    # Analytic strike for |delta| = target (r ~ 0): d1 = N^-1(delta) for calls, -N^-1(delta) for puts
    z = _ninv(target) if opt == "C" else -_ninv(target)
    k0 = f * math.exp(-z * sv + 0.5 * sv * sv)
    base = round(k0 / tick) * tick
    grid = [base + i * tick for i in range(-10, 11) if base + i * tick > 0]
    return min(grid, key=lambda k: abs(abs(delta(f, k, t, vol, opt, r)) - target))


def _ninv(p: float) -> float:
    """Inverse standard normal CDF via bisection (accurate enough for strike search)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _ncdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
