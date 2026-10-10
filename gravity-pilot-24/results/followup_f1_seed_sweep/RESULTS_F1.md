# Follow-up F1 results — 101 random 16-D projections (pre-specified in `protocol/FOLLOWUP_F1_SEED_SWEEP.md`)

Exploratory and descriptive. Same 24 frozen embeddings, same readers/folds/λ; only the random projection seed varies (pilot seed 20261009 + 20261010…20261109). The row for seed 20261009 reproduces the pilot exactly (validity check passed). The pilot's primary result is unchanged.

| | Value |
|---|---|
| Seeds passing the primary rule (E_proj < 16.53 m **and** ρ ≤ 1.10) | **1 of 101** (0.99 %) |
| Seeds with E_proj below the baseline bar (16.53 m) | 13 of 101 |
| E_projected, min / 25 % / median / 75 % / max | 11.99 / 18.07 / **20.20** / 22.97 / 33.58 m |
| ρ = E_proj / E_full (E_full = 12.17 m), same quantiles | 0.99 / 1.48 / **1.66** / 1.89 / 2.76 |
| Pilot seed 20261009 | E 22.32 m, ρ 1.83; 70th of 101 in ascending error (somewhat worse than median) |
| Groups (of 6) beating both baselines: median / max | 2 / 6 |
| Colour correct (of 24): min / median / max; seeds at 24/24 | 20 / 24 / 24; 83 of 101 |
| J_projected (colour sensitivity of position), median (min–max) | 11.9 m (7.4–20.7) vs full 4.5 m |

## Reading (pre-stated rule: few passes ⇒ typical failure)
* The original failure is **typical of random 16-D projections of these embeddings**, not a bad draw of seed 20261009: the median projected reader (20.2 m) is worse than the training-prior baseline (19.6 m) and 66 % above the full reader; about 1 seed in 8 gets under the baseline bar, and one seed in 101 also meets the 10 % tolerance.
* That single passing seed is **not** a result and is not selected or reported as "best": with 101 draws from a broad distribution a few extreme values are expected, and the pilot's decision rules were fixed for one seed.
* Colour stays highly readable from random 16-D projections (83/101 seeds perfect, worst 20/24), so the pilot's "no selective colour loss" also looks typical.
* Unchanged limits: same 24 clips and six start-position groups, dependent folds, near-interpolating full reader; this sweep says nothing about other clips, other dimensions or non-random (learned/PCA) projections, and carries no significance claim.

Files: `seed_sweep.csv` (all 101 rows), `summary.json`, `plot_f1_seed_sweep.png`; code `seed_sweep.py` (no model inference).
