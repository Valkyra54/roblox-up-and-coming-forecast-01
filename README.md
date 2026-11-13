# Roblox Up-and-Coming forecast

Predicts Roblox's Up-and-Coming top 10 over the next 72 hours, as a
per-place probability distribution over games.

## Why not persistence?

The chart churns violently: the game ranked #1 at the end of the practice
history is #1 in **none** of the 10 practice-window reads, and a median of
6 of the top 10 change between consecutive reads. The house baseline
(persist the last chart) scores **-43.51** on the practice window. A
snapshot cannot win here — the same forecast is scored against every
hourly read of the window, so what matters is the **time-averaged
occupancy** of each place over the whole 72 hours.

## Model — seeded Monte Carlo of latent log-rank

1. **Empirical rank transitions.** From every consecutive pair of reads in
   the history, the per-hour change in log-rank is fitted per rank band
   (1–3, 4–10, 11–30, 31–100): a mean drift and a volatility. Rank changes
   are heavy in the tail and the top band is no stickier than the rest, so
   no mean reversion or stickiness is imposed — both were tried and both
   made the backtest worse.
2. **Recent velocity.** Each game's log-rank velocity over its last five
   sightings (clipped to ±0.02/hour) is carried into the simulation as
   drift — a climber keeps climbing, a fader keeps fading.
3. **Simulation.** A seeded random walk (fixed seed, 1000 paths) of each
   game's latent score (−log rank) over the window's hours, re-sorting
   the chart every hour and counting per-place occupancies. Games unseen
   for >96 hours are dropped; a game whose `details.created` is more than
   28 days before the window start is excluded, and one that ages out
   mid-window leaves the simulation at that hour (archive reads carry no
   `created`, so this degrades gracefully to "no exit").
4. **Probabilities.** Occupancies are mildly flattened (tempering
   exponent 0.7) against overconfidence, then normalised per place to a
   named mass of 0.75 — the remaining **0.25 is left unnamed**, the only
   hedge for brand-new games joining the chart (in the practice window a
   game that was not in the chart at all took #1 in three of ten reads).
   `top10` is the ranking by expected place under the same occupancies.

## Backtest (practice window, 2026-11-08 → 2026-11-10, 10 reads)

Trained on the 34-read history up to 2026-11-07T23:00Z, scored against
the 10 practice-window reads reconstructed from the public archive.

| model | log score | typical p per place | mean top-10 hits |
| --- | --- | --- | --- |
| house baseline (persistence) | **-43.51** | 0.0129 | 6.4 |
| recency-decayed rank distribution | -39.06 | 0.0201 | 6.1 |
| direct decayed occupancy | -43.54 | 0.0129 | 4.4 |
| **this entry (Monte Carlo)** | **-34.31** | **0.0323** | **6.1** |
| oracle (answer-derived ceiling) | -15.41 | 0.2142 | 7.4 |

Verified on the shipped `main.py` (1000 paths, seed fixed): two runs
byte-identical, 18 s runtime, output 10,091 bytes, sha256
`49d4a838edb0af97dfd54c319a9022caa931e3afb472e47c9fd667cb3baa9b68`.
Per-read scores run from -23.9 (first read) to -42.8 (last read) — the
forecast degrades as the window extends, exactly as a time-averaged
model should when the chart keeps churning.

The Monte Carlo beats the baseline by ~9 log points. It does not reach
the -25 target: the remaining gap is mostly unforecastable — brand-new
entrants taking #1 (Do Not Stop, absent from the history entirely, was
#1 in 3 of 10 reads) and the order shuffling among a stable core set.
`playing` growth was tested as a signal and is useless (correlation
-0.01 with rank change), as is Top Trending rank (correlation 0.02 with
rank change). The final history will be denser (hourly house reads, not
the sparse archive), which should sharpen the fitted volatilities and
velocities beyond what this backtest shows.

## Files

- `main.py` — the entry program (Python 3.13, stdlib only, no network).
  Reads `$MAKEFUN_INPUT_DIR` (default `/input`), writes
  `$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output`).
- `LICENSE` — MIT.
