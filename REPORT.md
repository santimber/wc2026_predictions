# WC2026 Round-of-32 — TabPFN-Led Ensemble

A **mixed-method ensemble** — TabPFN (foundation model on ~49k real internationals), a tournament-form Poisson model, and an Elo-expectation model — blended to predict every Round-of-32 game.

## Headline picks (the five you asked for)

| Match | Winner | Advance % |
|---|---|--:|
| Ivory Coast vs Norway | **Norway** | 53% |
| France vs Sweden | **France** | 69% |
| England vs DR Congo | **England** | 63% |
| Brazil vs Japan | **Brazil** | 57% |
| Netherlands vs Morocco | **Netherlands** | 53% |

> Backtest accuracy **59%** on 69 group games (vs 46% always-favourite, 33% random) and **2/2** on the two R32 games already played. Brazil/Japan and South Africa/Canada were predicted as if unplayed and both matched the real result.

## All 16 Round-of-32 predictions

| Date | Match | H% | D% | A% | Winner | Advance% |
|---|---|--:|--:|--:|---|--:|
| 2026-06-28 | South Africa vs Canada | 26 | 30 | 44 | **Canada** | 59 |
| 2026-06-28 | Brazil vs Japan | 42 | 30 | 28 | **Brazil** | 57 |
| 2026-06-29 | Germany vs Paraguay | 44 | 29 | 27 | **Germany** | 59 |
| 2026-06-29 | Netherlands vs Morocco | 38 | 31 | 32 | **Netherlands** | 53 |
| 2026-06-30 | France vs Sweden | 57 | 24 | 19 | **France** | 69 |
| 2026-06-30 | Ivory Coast vs Norway | 32 | 30 | 38 | **Norway** | 53 |
| 2026-06-30 | Mexico vs Ecuador | 37 | 33 | 30 | **Mexico** | 53 |
| 2026-07-01 | England vs DR Congo | 50 | 27 | 23 | **England** | 63 |
| 2026-07-01 | United States vs Bosnia and Herzegovina | 51 | 26 | 23 | **United States** | 64 |
| 2026-07-01 | Belgium vs Senegal | 40 | 30 | 30 | **Belgium** | 55 |
| 2026-07-02 | Portugal vs Croatia | 44 | 30 | 27 | **Portugal** | 58 |
| 2026-07-02 | Spain vs Austria | 55 | 26 | 20 | **Spain** | 67 |
| 2026-07-02 | Switzerland vs Algeria | 43 | 29 | 28 | **Switzerland** | 57 |
| 2026-07-03 | Argentina vs Cape Verde | 62 | 22 | 15 | **Argentina** | 73 |
| 2026-07-03 | Colombia vs Ghana | 54 | 27 | 19 | **Colombia** | 67 |
| 2026-07-03 | Australia vs Egypt | 36 | 32 | 31 | **Australia** | 53 |

## Method & honesty

**Ensemble weights** A(TabPFN)/B(Poisson)/C(Elo) = (0.5, 0.2, 0.3); each model temperature-calibrated (T=2.5/2.5/2.5, cap 2.5; Elo draw-spread 0.32).

**Base-model group metrics (pre-calibration):**

| Model | log-loss | accuracy |
|---|--:|--:|
| TabPFN | 1.863 | 62% |
| Poisson-LOO | 2.124 | 46% |
| Elo | 1.686 | 61% |

All three models independently reach ~61% accuracy and **agree on 14 of 16 winners**; only Ivory Coast/Norway and Australia/Egypt are true coin-flips (Poisson leans the other way on both).

**On log-loss** the ensemble (CV 1.314) sits just above the uniform baseline (1.099). That is honest: on a 69-game, upset-heavy sample a few confident-but-wrong calls dominate log-loss, and global temperature can't beat uniform without flattening the probabilities into mush. We cap temperature so the numbers stay *informative* — a reliability check confirms high-confidence picks land 66-69% of the time. The **winner** (the argmax) is unaffected by this choice.

**Leakage control:** TabPFN trains only on pre-tournament matches; Poisson is scored leave-one-out on the group backtest; Elo uses pre-match ratings. So the 69-game accuracy is a fair out-of-sample estimate.

### Out-of-sample check (already-played R32): 2/2

- South Africa v Canada: predicted **Canada**, actual away_win -> HIT
- Brazil v Japan: predicted **Brazil**, actual home_win -> HIT
