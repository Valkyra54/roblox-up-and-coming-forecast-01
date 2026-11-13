# roblox-up-and-coming-forecast-01 — time-averaged occupancy forecast.
#
# The same forecast is scored against EVERY hourly read of a 72-hour window,
# so per-place probabilities must be the TIME-AVERAGED occupancy over the
# window, not a snapshot of the last chart. The chart churns violently
# (the #1 at history end is #1 in none of the 10 practice-window reads),
# so persistence scores -43.5. This program instead:
#   1. fits an empirical rank-transition model from every consecutive pair
#      of reads (per-hour drift and volatility of log-rank, by rank band);
#   2. runs a SEEDED Monte Carlo of the window's hours: each game's latent
#      score (-log rank) random-walks with its recent velocity, ranks are
#      re-sorted every hour, and per-place occupancies are counted;
#   3. mildly flattens the occupancies (tempering) against overconfidence,
#      caps the named mass per place, and leaves a generous unnamed share
#      (the only hedge for brand-new games joining the chart).
# Stdlib only, no network, deterministic (fixed seed, no wall clock).
# Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
# (default /output). Window length, places and output name come from
# task.json — never hardcoded.

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
N_PATHS = 1000
UNNAMED = 0.25          # mass left unnamed at each place (new-game hedge)
MAX_NAMED = 60          # rows named per place at most
TEMPER_ALPHA = 0.7      # flattening exponent against overconfidence
VEL_CLIP = 0.02         # max |log-rank velocity| per hour
GAP_PENALTY = 0.01      # score penalty per hour since a game was last seen
MAX_ABSENT_HOURS = 96   # ignore games unseen for longer than this
AGE_DAYS = 28           # games leave the chart when created > 28 days ago


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def band_of(rank):
    if rank <= 3:
        return "1-3"
    if rank <= 10:
        return "4-10"
    if rank <= 30:
        return "11-30"
    return "31-100"


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    return task, reads


def created_map(reads):
    """universe_id -> created datetime, from details when present (may be null)."""
    created = {}
    for read in reads:
        details = read.get("details")
        if not details:
            continue
        items = []
        if isinstance(details, dict):
            for key, value in details.items():
                items.append((key, value))
        elif isinstance(details, list):
            for value in details:
                if isinstance(value, dict) and "universe_id" in value:
                    items.append((value["universe_id"], value))
        for key, value in items:
            if not isinstance(value, dict):
                continue
            try:
                uid = int(key)
            except (TypeError, ValueError):
                continue
            made = value.get("created")
            if made and uid not in created:
                try:
                    created[uid] = parse_time(made)
                except ValueError:
                    pass
    return created


def fit_volatility(charts):
    """Per rank band: (mean, std) of per-hour change in log-rank."""
    samples = {"1-3": [], "4-10": [], "11-30": [], "31-100": []}
    for i in range(len(charts) - 1):
        t0, c0 = charts[i]
        t1, c1 = charts[i + 1]
        dt = (t1 - t0).total_seconds() / 3600.0
        if dt <= 0:
            continue
        inv1 = {uid: rank for rank, uid in c1.items()}
        for rank, uid in c0.items():
            rank2 = inv1.get(uid)
            if rank2 is None:
                continue
            samples[band_of(rank)].append(
                (math.log(rank2) - math.log(rank)) / dt
            )
    vol = {}
    for band, values in samples.items():
        if len(values) >= 5:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            vol[band] = (mean, max(math.sqrt(var), 0.02))
        else:
            vol[band] = (0.0, 0.05)
    return vol


def build_games(charts, t_end, created, window_start):
    """uid -> [score, velocity]; score is -log(rank) at the last sighting."""
    observations = {}
    for when, chart in charts:
        hours = (when - charts[0][0]).total_seconds() / 3600.0
        for rank, uid in chart.items():
            observations.setdefault(uid, []).append((hours, rank))
    total_hours = (t_end - charts[0][0]).total_seconds() / 3600.0
    games = {}
    for uid, obs in observations.items():
        last_h, last_rank = obs[-1]
        gap_end = total_hours - last_h
        if gap_end > MAX_ABSENT_HOURS:
            continue
        # 28-day aging exit: drop games already too old at the window start.
        made = created.get(uid)
        if made is not None and made + timedelta(days=AGE_DAYS) <= window_start:
            continue
        recent = obs[-5:]
        velocity = 0.0
        if len(recent) >= 2:
            span = recent[-1][0] - recent[0][0]
            if span > 0:
                velocity = (
                    math.log(recent[-1][1]) - math.log(recent[0][1])
                ) / span
        velocity = max(-VEL_CLIP, min(VEL_CLIP, velocity))
        score = -math.log(last_rank) - GAP_PENALTY * gap_end
        games[uid] = [score, velocity, made]
    return games


def run_monte_carlo(games, vol, n_hours, window_start, places):
    """Seeded simulation; returns {(uid, place): occupancy probability}."""
    uids = sorted(games)
    start_score = {uid: games[uid][0] for uid in uids}
    # Hour (0-based) at which each game ages out of the chart, if any.
    exit_hour = {}
    for uid in uids:
        made = games[uid][2]
        if made is not None:
            exit_at = made + timedelta(days=AGE_DAYS)
            hour = (exit_at - window_start).total_seconds() / 3600.0
            if 0 <= hour < n_hours:
                exit_hour[uid] = int(math.ceil(hour))
    rng = random.Random(SEED)
    counts = {}
    for _ in range(N_PATHS):
        scores = dict(start_score)
        for step in range(n_hours):
            for uid, hour in exit_hour.items():
                if step >= hour:
                    scores[uid] = -1e18
            order = sorted(uids, key=lambda u: (-scores[u], u))
            for place in range(min(places, len(order))):
                key = (order[place], place + 1)
                counts[key] = counts.get(key, 0) + 1
            for uid in uids:
                if scores[uid] <= -1e17:
                    continue
                est_rank = max(1, min(100, round(math.exp(-scores[uid]))))
                mean, sigma = vol[band_of(est_rank)]
                scores[uid] += -games[uid][1] + rng.gauss(-mean * 0.3, sigma)
    total = N_PATHS * n_hours
    return {key: value / total for key, value in counts.items()}


def build_forecast(task, reads):
    places = int(task.get("places", 10))
    window = task.get("window", {})
    n_hours = int(window.get("slots", 72))
    window_start = parse_time(window["from"]) if window.get("from") else None
    charts = []
    for read in reads:
        rows = read.get("up_and_coming") or []
        chart = {}
        for row in rows:
            try:
                chart[int(row["rank"])] = int(row["universe_id"])
            except (KeyError, TypeError, ValueError):
                continue
        if chart:
            charts.append((parse_time(read["at"]), chart))
    if not charts:
        return {"probabilities": [], "top10": []}
    charts.sort(key=lambda item: item[0])
    t_end = charts[-1][0]
    if window_start is None:
        window_start = t_end
    created = created_map(reads)
    vol = fit_volatility(charts)
    games = build_games(charts, t_end, created, window_start)
    if not games:
        return {"probabilities": [], "top10": []}
    probs = run_monte_carlo(games, vol, n_hours, window_start, places)
    rows = []
    for place in range(1, places + 1):
        cand = sorted(
            ((p, uid) for (uid, k), p in probs.items() if k == place),
            reverse=True,
        )[:MAX_NAMED]
        total = sum(p for p, _ in cand)
        if total <= 0:
            continue
        tempered = [(p ** TEMPER_ALPHA, uid) for p, uid in cand]
        tsum = sum(w for w, _ in tempered)
        for weight, uid in tempered:
            rows.append(
                {
                    "universe_id": uid,
                    "place": place,
                    "p": round(weight / tsum * (1.0 - UNNAMED), 6),
                }
            )
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    totals = {}
    expected = {}
    for (uid, place), p in probs.items():
        expected[uid] = expected.get(uid, 0.0) + place * p
        totals[uid] = totals.get(uid, 0.0) + p
    ranked = sorted(
        ((expected[uid] / totals[uid], uid) for uid in totals if totals[uid] > 0),
        key=lambda item: (item[0], item[1]),
    )
    top10 = [uid for _, uid in ranked[:places]]
    if len(top10) < places:
        for _, uid in sorted(charts[-1][1].items()):
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break
    return {"probabilities": rows, "top10": top10[:places]}


def main():
    task, reads = load_history(INPUT_DIR)
    forecast = build_forecast(task, reads)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
