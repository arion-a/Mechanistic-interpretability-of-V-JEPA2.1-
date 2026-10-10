# Results — 24-video gravity pilot (protocol v1.0 maths, dataset v1.1, CPU, one run)

**Descriptive pilot. 24 clips, 6 start-position groups, dependent folds. No significance claims, no uncertainty estimated.** All numbers are reproduced from `predictions.csv` (30 values re-computed and matched by `stage_b.py report`). Settings were fixed before running: checkpoint `vjepa2_1_vitb_dist_vitG_384.pt`, final post-LayerNorm tokens, seed 20261009, 16-D projection, λ = 0.01. Nothing was re-run to change an outcome; three complete runs gave bit-identical embeddings, predictions and summary.

## Headline (decision rules from protocol §9)
| Quantity | Value |
|---|---|
| E_full (Euclidean RMSE, 24 out-of-fold clips) | **12.17 m** |
| E_projected | **22.32 m** |
| E_CV (oracle constant-velocity baseline) | 16.53 m |
| E_prior (training-prior baseline) | 19.60 m |
| Bar = min(E_CV, E_prior) | 16.53 m |
| G_full = 1 − E/E_CV ; G_projected | +0.264 ; −0.350 |
| ρ = E_projected / E_full | **1.834** (tolerance ≤ 1.10); absolute change +10.15 m |
| **Primary prediction-retention result** | **NOT met** — full beats the bar, projected does not, ρ > 1.10 |
| Colour correct (full / projected) | **24/24 / 24/24**; ΔA = 0 |
| Selective-accessibility pattern (A_full > 0.5 and ΔA ≥ 0.125) | **Not present** (ΔA = 0) |
| Groups beating both baselines (b_r) | full 5/6 (fails scene 3); projected 1/6 (scene 1 only) |
| Storage | 3,072 → 64 B per clip (48×); 73,728 → 1,536 B for 24 clips; + 49,152 B projection matrix = 50,688 B |

## Per held-out group (m)
| Scene | Full | Projected | CV | Prior | ρ_f |
|---|---|---|---|---|---|
| 0 | 7.03 | 19.09 | 16.53 | 19.6 | 2.71 |
| 1 | 9.44 | 9.86 | 16.53 | 19.6 | 1.05 |
| 2 | 14.11 | 24.66 | 16.53 | 19.6 | 1.75 |
| 3 | 17.30 | 24.43 | 16.53 | 19.6 | 1.41 |
| 4 | 7.26 | 17.26 | 16.53 | 19.6 | 2.38 |
| 5 | 14.09 | 31.99 | 16.53 | 19.6 | 2.27 |

## Declared diagnostics
* Per coordinate (pooled): full x 8.50 m, y 8.71 m; projected x 11.84 m, y 18.91 m; CV x ≈ 0, y 16.53 m; prior x 0, y 19.60 m.
* Per gravity (pooled): full 12.34 m (g=9.8), 11.99 m (g=4.9); projected 18.36 / 25.67; CV 20.91 / 10.45. The pooled bar hides that the full reader is **worse than the CV oracle at g=4.9** and better at g=9.8.
* Paired colour sensitivity J (shift of predicted position when only colour changes): full 4.54 m, projected 14.73 m.
* Colour confusion matrices are perfect (12/12 orange, 12/12 blue) for both representations in every group.

## What this does and does not say
* **Supported (narrow, descriptive):** under these fixed readers, the full 768-D mean-pooled embedding of the first 2 s gave lower future-position error than both baselines pooled and in 5 of 6 held-out start positions. The fixed seeded 16-D random projection did not retain that: its error exceeds both baselines and is 83 % above the full reader, far outside the 10 % tolerance. Colour was perfectly readable from both representations.
* **Therefore:** the pre-specified compression-retention claim is not met, and the colour result shows **no** selective colour loss. Colour did not deteriorate, so the "both deteriorate" reading does not apply; the observed pattern is projection hurting position much more than colour.
* **Not supported:** this is not evidence that JEPA understands gravity or physics, that colour is "abstracted away" (colour was fully accessible), or that 16 dimensions can never work — it is one random projection (no PCA/learned subspace by design). Position error of the full reader is still large (x error 8.5 m although x is a simple function of visible start position), the full reader is near-interpolating (20 training clips, 768 features, λ = 0.01), a ground shadow is an extra horizontal-position cue, colour is not luminance-matched, and 24 clips in 6 dependent groups cannot support significance claims.
* Colour accessibility at 24/24 means a ball colour difference is easy to read even after random projection; that says little about abstraction.

Files: `predictions.csv`, `summary.json`, `provenance.json`, `embeddings_*.npy`, `projection.npy`, `embedding_index.csv`, `fitted_readers/` (24 fits = 6 folds × 2 representations × 2 tasks, all means/scales/weights/split IDs), `plot1–4_*.png` with raw data in `plot_data/`.
