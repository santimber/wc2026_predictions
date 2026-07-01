# WC2026 Round-of-32 — TabPFN-Led Ensemble

A **mixed-method ensemble** that predicts every 2026 World Cup Round-of-32 game,
built on Prior Labs' [TabPFN football template](https://ux.priorlabs.ai/worldcup).

Three models are temperature-calibrated and blended:

| Model | Trained on | Captures | Weight |
|---|---|---|--:|
| **A · TabPFN** | ~1000 recent real internationals (leakage-free ELO / form / h2h / rest features), with the completed WC2026 group stage merged in | Deep real-world team strength | 0.50 |
| **B · Poisson** | WC2026 tournament attack/defense indices + pre-tournament prior | This tournament's scoring form | 0.20 |
| **C · Elo** | Pre-match Elo expectation with a draw spread | Calibrated match expectation | 0.30 |

TabPFN is a foundation model for small tabular data — an ideal fit for football
outcome prediction. It carries the most weight (best discrimination); Poisson and
Elo add tournament-specific signal and calibration.

## Headline result

- **59% accuracy** on a 69-game group-stage backtest (vs 46% always-favourite,
  33% random), and **2/2** on the two R32 games already played — Brazil beat
  Japan and Canada beat South Africa were both called correctly, predicted as if
  unplayed.
- Full picks and probabilities are in **[REPORT.md](REPORT.md)** and
  **[predictions_R32.csv](predictions_R32.csv)** (competition schema:
  `date, home_team, away_team, predicted, p_home_win, p_draw, p_away_win`).

The five requested knockouts: **France** over Sweden, **England** over DR Congo,
**Brazil** over Japan, **Netherlands** over Morocco, and — the one upset —
**Norway** over Ivory Coast.

## How it works

```
data assembly ──► Model A (TabPFN)  ┐
(real history +   Model B (Poisson) ├─ temperature-calibrate ─► weight-blend ─► predictions
 WC2026 results)  Model C (Elo)     ┘
```

- `predict_ensemble.py` — merges the real international dataset (`results.csv`,
  ~49k matches) with the completed WC2026 group results in `data/`, builds
  leakage-free features, and defines the base models.
- `fit_worker.py` — one isolated TabPFN fit (run as a killable subprocess; the
  hosted API is latency-variable, so hangs are timed out and retried).
- `lab.py` — orchestrates several TabPFN training-pool configs, caches each, and
  is fully resumable.
- `calibrate.py` — **the final pipeline**: temperature-calibrates each model,
  blends them, and writes `predictions_R32.csv` + `REPORT.md`.
- `analyze.py` — diagnostics (reliability curves, weight-scheme comparison).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce the final predictions from the cached TabPFN output (no API needed):
python calibrate.py

# Or refit TabPFN from scratch (needs a TABPFN_TOKEN env var or browser login):
python lab.py && python calibrate.py
```

`cache/A_comp1000.pkl` holds TabPFN's cached predictions so the final numbers
reproduce offline. `feats_cache.pkl` (the feature matrix) is regenerated
automatically on first run.

## Honesty notes

- **Log-loss** sits just above the uniform baseline (1.099). On a 69-game,
  upset-heavy sample a few confident-but-wrong calls dominate log-loss, and
  global temperature can't beat uniform without flattening probabilities into
  mush. Temperature is capped so the numbers stay informative — a reliability
  check confirms high-confidence picks land 66–69% of the time. The **winner**
  (the argmax) is unaffected.
- **Leakage control:** TabPFN trains only on pre-tournament matches; Poisson is
  scored leave-one-out on the backtest; Elo uses pre-match ratings. The 69-game
  accuracy is a fair out-of-sample estimate.
- Tournament form rests on ~3 games per team, so Models B/C are noisy. Not a
  betting model.

Data: [martj42/international_results](https://github.com/martj42/international_results)
(real internationals) + openfootball WC2026 results.
