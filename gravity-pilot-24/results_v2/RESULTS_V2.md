# Results — Stage B on dataset v2.0 (confound-reduced, 48 clips, 12 start positions, CPU, one run)

Pre-specified in `protocol/STAGE_B_V2_PRESPEC.md` (committed before any v2 embedding existed). Same encoder, layer, projection (seed 20261009), readers, λ = 0.01 and decision rules as the pilot; only counts changed (48 clips, 12 leave-one-position-out folds, 44 train / 4 test). Descriptive; no significance claims; 54 summary values re-derived from `predictions.csv` and matched. Approval receipt: manifest `75ea34ed…b6ed`, 2026-10-10T09:05:22Z.

## Headline (decision rules from protocol §9, unchanged)
| Quantity | v2.0 (this run) | Pilot v1.1 (reference) |
|---|---|---|
| E_full (Euclidean RMSE, out-of-fold) | **15.93 m** | 12.17 m |
| E_projected (16-D) | **28.33 m** | 22.32 m |
| E_CV / E_prior / bar = min | 16.53 / 19.60 / 16.53 m | same |
| G_full = 1 − E_full/E_CV | **+0.036** | +0.264 |
| ρ = E_proj / E_full (tolerance ≤ 1.10) | **1.778** (+12.40 m) | 1.834 |
| **Primary prediction-retention rule** | **NOT met** (full barely under the bar; projected far above; ρ > 1.10) | not met |
| Colour correct, full / projected | **48/48 / 48/48**, ΔA = 0 | 24/24 / 24/24 |
| Selective-accessibility pattern | **Not present** | not present |
| Held-out positions beating both baselines | full **7/12**, projected **0/12** | 5/6 and 1/6 |
| J (colour sensitivity of predicted position), full / projected | 7.16 / 17.62 m | 4.54 / 14.73 m |
| Storage | 3,072 → 64 B per clip (48×); 48 clips 147,456 → 3,072 B; + 49,152 B matrix = 52,224 B | 24 clips, 73,728 → 1,536 B; 50,688 B |

## Per held-out start position (m)
| Pos | x0,y0 | Full | Projected | CV | Prior | ρ_f | Full beats both? |
|---|---|---|---|---|---|---|---|
| 0 | −60,120 | 22.63 | 32.39 | 16.53 | 19.6 | 1.43 | no |
| 1 | −20,120 | 13.60 | 18.84 | 16.53 | 19.6 | 1.38 | yes |
| 2 | 20,120 | 5.34 | 16.98 | 16.53 | 19.6 | 3.18 | yes |
| 3 | 60,120 | 14.42 | 17.78 | 16.53 | 19.6 | 1.23 | yes |
| 4 | −60,140 | 15.98 | 35.28 | 16.53 | 19.6 | 2.21 | yes |
| 5 | −20,140 | 6.34 | 17.65 | 16.53 | 19.6 | 2.78 | yes |
| 6 | 20,140 | 7.10 | 19.06 | 16.53 | 19.6 | 2.69 | yes |
| 7 | 60,140 | 18.73 | 46.35 | 16.53 | 19.6 | 2.47 | no |
| 8 | −60,160 | 19.09 | 31.46 | 16.53 | 19.6 | 1.65 | no |
| 9 | −20,160 | 19.35 | 29.22 | 16.53 | 19.6 | 1.51 | no |
| 10 | 20,160 | 11.72 | 18.48 | 16.53 | 19.6 | 1.58 | yes |
| 11 | 60,160 | 23.23 | 36.99 | 16.53 | 19.6 | 1.59 | no |

## Declared diagnostics
* Per coordinate (pooled): full x 12.96 m, y 9.26 m; projected x 23.75, y 15.44; CV x ≈ 0, y 16.53; prior x 0, y 19.60. (Pilot full: x 8.50, y 8.71.)
* Per gravity: full 16.68 m (g = 9.8; CV 20.91) and 15.15 m (g = 4.9; CV **10.45**); projected 28.85 / 27.80. At g = 4.9 the full reader is worse than the constant-velocity oracle.
* Colour confusion matrices perfect (24/24 each colour, both representations, all 12 groups).

## Reading
* **Primary result is again "not met", for the same reason:** the 16-D projection loses the position signal (28.3 m, 78 % above the full reader, above both baselines in all 12 positions). Colour is perfectly readable from both, so there is **no selective colour loss**.
* **The full-embedding result is weaker than the pilot suggested.** With the shadow cue removed, colours luminance-matched and twice as many held-out positions, E_full rose from 12.2 to 15.9 m and its margin over the constant-velocity oracle fell from 26 % to 3.6 %: it beats both baselines in only 7 of 12 positions, loses at low gravity, and its x error grew most (8.5 → 13.0 m). So "the full embedding carries future-position information under a linear reader" is supported only marginally in this harder, cleaner setting.
* **Attribution is not possible:** shadow removal, luminance matching and the new 12-position fold structure were changed together, and the positions themselves differ (new x0, y0 values, more extreme starts). The larger x error is consistent with the shadow having been a horizontal-position cue in v1.1, but this run cannot separate that from the other changes. No pooled test of v1.1 vs v2.0 is made; they are different stimuli.
* **Not supported:** that JEPA understands gravity, that colour is abstracted away (colour 48/48), or anything about non-random projections. Same limits as before: dependent folds, near-interpolating full reader (44 clips vs 768 features), colour hue/saturation still confounded with the label by design, single projection seed here (the 101-seed sweep on v1.1 suggests the projection failure is typical, but it was not repeated on v2).

Files (`results_v2/`): `predictions.csv`, `summary.json`, `provenance.json`, `embeddings_*.npy`, `projection.npy`, `embedding_index.csv`, `fitted_readers/` (48 fits = 12 folds × 2 representations × 2 tasks), `plot1–4_*.png` + `plot_data/`.
