"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model (stdlib only, deterministic, no network):

An earn rate is a ratio of two growing counts: the players who earned a
trophy, E(t), over the players Exophase tracks, N(t).  Both counts grow
roughly like powers of time since release, so on the log-time axis
x = ln(1 + hours since 2026-11-19T00:00Z) each trophy's logit rate is
roughly linear in x.  For every trophy we fit that line by least squares
over the case's reads, shrink its slope toward the median slope of its
tier (platinum / gold / silver / bronze), anchor the line at the last
read's rate, and read it at the target.  The spread combines a
random-walk term that grows with the horizon, the slope's standard error
times the extrapolation distance, and extra width when the page has gone
stale since its last change.

One special regime matters: a page Exophase has not refreshed for hours
shows a completely flat history (every read identical).  A refresh
catches the tracked-player count up in one jump while earned counts
move much less, so rates fall sharply -- on launch night the first
refresh took players tracked 638 -> 3,868 while earned counts grew only
~1.34x, so non-100% rates fell to ~0.22x.  When the whole history is
flat and the horizon is short (a refresh is imminent), the median is
shifted down by 0.70 * sqrt(frozen_hours + horizon_hours) logit units,
capped at -2.5, with a wide spread.  A trophy already at 100% is left
at 100%: it survived a 6x influx unchanged.

Usage (as the grader runs it): python3 -P -s /bot/main.py, reading
every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writing
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Logit-scale spread per sqrt(day) of horizon (random-walk term).
RW_SIGMA = 0.55
# Down-shift per sqrt(hour) of frozen+ horizon time in the flat-page regime.
FLAT_SHIFT = 0.70
FLAT_SHIFT_CAP = 2.5
FLAT_SIGMA = 1.0


def parse_time(text):
    """ISO 8601 -> aware datetime; accepts a trailing Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def log_time(moment):
    return math.log1p(hours_since_release(moment))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, se_b, n_distinct_x).

    se_b is None when the slope is not identified (too few distinct x
    or no residual degrees of freedom).
    """
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    se_b = None
    if n >= 3:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        dof = n - 2
        if dof > 0 and rss >= 0:
            se_b = math.sqrt((rss / dof) / sxx)
    return (a, b, se_b, len(set(xs)))


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = log_time(target)

    # Fallback when there is no history at all: flat 1% with a wide band.
    if not history:
        rows = []
        for trophy in trophies:
            centre = logit(1.0)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(centre + z * 1.5)) for z in Z]})
        return {"trophies": rows}

    xs = [log_time(parse_time(read["at"])) for read in history]
    x_last = xs[-1]
    n_tracked = [read["players_tracked"] for read in history]

    # Is the page frozen?  Every read identical in players and rates.
    flat_page = (
        len(set(n_tracked)) == 1
        and all(read["rates"] == history[0]["rates"] for read in history)
    )
    frozen_hours = (checkpoint - parse_time(history[0]["at"])).total_seconds() / 3600.0

    # Per-trophy logit series and OLS slopes.  Reads at the 0.01/99.99
    # clamps carry no slope information (a trophy pinned at 100% looks
    # like a cliff when it first moves), so the fit uses interior rates
    # only.  The slope blends the full history with the last 24 hours,
    # so a launch-night regime change does not drag the extrapolation.
    slopes = []
    tier_slopes = {}
    per_trophy = []
    for i, trophy in enumerate(trophies):
        series = []
        for j, read in enumerate(history):
            rate = read["rates"][i]
            if rate is not None:
                series.append((xs[j], logit(rate)))
        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if last_rate is None:
            last_rate = 1.0
        interior = [(x, y) for (x, y) in series
                    if 0.01 < inv_logit(y) < 99.99]
        fit = ols([p[0] for p in interior], [p[1] for p in interior]) if len(interior) >= 2 else None
        slope = fit[1] if fit else 0.0
        se_b = fit[2] if fit else None
        n_distinct = fit[3] if fit else 0
        # Blend with the last-24h slope so a regime change (e.g. a
        # trophy leaving the 100% plateau) does not drag the forecast.
        x_cut = log_time(checkpoint - timedelta(hours=24))
        recent_pts = [(x, y) for (x, y) in interior if x >= x_cut]
        recent_fit = (ols([p[0] for p in recent_pts], [p[1] for p in recent_pts])
                      if len(recent_pts) >= 3 else None)
        if recent_fit is not None:
            slope = 0.5 * slope + 0.5 * recent_fit[1]
            if se_b is None or (recent_fit[2] is not None and recent_fit[2] > se_b):
                se_b = recent_fit[2]
        slopes.append(slope)
        tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(slope)
        per_trophy.append({
            "id": trophy["id"],
            "tier": trophy.get("tier", "bronze"),
            "last_rate": last_rate,
            "slope": slope,
            "se_b": se_b,
            "n_distinct": n_distinct,
            "series": series,
        })

    global_slope = median(slopes) if slopes else 0.0
    tier_median = {}
    for tier, vals in tier_slopes.items():
        tier_median[tier] = median(vals) if len(vals) >= 2 else global_slope

    # Hours since this trophy's rate last changed (staleness).
    def stale_hours(info):
        series = info["series"]
        if len(series) < 2:
            return frozen_hours
        last_y = series[-1][1]
        for k in range(len(series) - 2, -1, -1):
            if abs(series[k][1] - last_y) > 1e-9:
                return (checkpoint - parse_time(history[k + 1]["at"])).total_seconds() / 3600.0
        return frozen_hours

    rows = []
    for info in per_trophy:
        last = info["last_rate"]
        if last >= 99.99:
            # Everybody tracked holds it; it survived a 6x influx at 100%.
            rows.append({"id": info["id"], "q": [99.99, 100, 100, 100, 100]})
            continue

        if flat_page and horizon_hours <= 24.0:
            shift = -min(FLAT_SHIFT * math.sqrt(max(0.0, frozen_hours + horizon_hours)),
                         FLAT_SHIFT_CAP)
            centre = logit(last) + shift
            sigma = FLAT_SIGMA
        else:
            # Shrink the per-trophy slope toward its tier median.
            prior = tier_median.get(info["tier"], global_slope)
            w = info["n_distinct"] / (info["n_distinct"] + 4.0) if info["n_distinct"] else 0.0
            slope = w * info["slope"] + (1.0 - w) * prior
            centre = logit(last) + slope * (x_target - x_last)
            # A stale last read lags reality: expect a small dilution
            # at the next refresh and widen the band for it.
            stale = stale_hours(info)
            if stale > 2.0:
                centre -= min(0.06 * math.sqrt(stale), 0.30)
            lever = max(0.0, x_target - x_last)
            sigma2 = (RW_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2
            if info["se_b"] is not None:
                sigma2 += (info["se_b"] * lever) ** 2
            else:
                sigma2 += (0.30 * lever) ** 2
            sigma = math.sqrt(sigma2)
            if stale > 2.0:
                sigma += min(0.10 * math.sqrt(stale), 0.40)
            sigma = min(max(sigma, 0.20), 2.0)

        centre = max(-9.0, min(9.0, centre))
        quantiles = [round4(min(99.99, max(0.01, inv_logit(centre + z * sigma))))
                     for z in Z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": info["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
