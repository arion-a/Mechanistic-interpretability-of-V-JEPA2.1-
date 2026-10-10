# Stage B on dataset v2.0 — pre-specification (written and committed BEFORE any v2 embedding exists)

Approval: receipt `dataset_v2/approval_receipt.json`, manifest SHA-256 `75ea34ed4ce6806bd1d065db80c76dcc6e39cc4422c0a548477813c4b0ebb6ed`, approved by the user 2026-10-10T09:05:22Z.

**Everything is the pilot's protocol v1.0 maths; only counts change because the dataset is larger.** Nothing below may be changed after seeing v2 results.

| Item | Fixed value |
|---|---|
| Encoder / layer / input | identical to the pilot: `vjepa2_1_vitb_dist_vitG_384.pt` (sha256 `848a77c3…ddf4d`), `ema_encoder`, final post-LayerNorm tokens, frames 0,4,…,60 only, float32, mean-pooled 768-D |
| Projection | **one** seed 20261009, 16-D, identical matrix to the pilot (sha256 `2a754eade0be27ac5840532226b95ad010cd849f130e1ada2459056ca1c6da2c`). No seed sweep on v2 unless separately requested |
| Readers | ridge, λ = 0.01, training-only mean + scalar RMS scale, dual solve, unpenalised intercept, targets p/100 and ∓1 colour, tie → orange |
| Folds | leave-one-**start-position**-out, **12 folds**, 44 train (22 orange / 22 blue, 22 per gravity) / 4 test; 4 readers per fold-representation pair → 12 × 2 × 2 = **48 readers** |
| Baselines | oracle constant-velocity (t_a = 56/30, t_b = 2) and training-prior; analytic values E_CV = 16.528 m, E_prior = 19.6 m (balanced training sets) |
| Primary rule | E_full < min(E_CV, E_prior) **and** E_projected < min(E_CV, E_prior) **and** ρ = E_proj/E_full ≤ 1.10 |
| Selective-accessibility pattern | A_full > 0.5 **and** ΔA = A_full − A_projected ≥ **0.125** (the pilot's threshold kept as a fixed number; with n = 48 this is 6/48) |
| Breadth | fraction of the 12 held-out start positions whose RMSE beats both baselines, for each representation; plus all 12 group RMSEs and per-fold ρ |
| Paired colour sensitivity J | 24 pairs (12 positions × 2 gravities), same formula |
| Storage | per clip 3,072 B vs 64 B (48×); 48 clips: 147,456 B vs 3,072 B; + projection matrix 49,152 B |
| Reporting | every metric, table and plot; descriptive only; no p-values/CIs; invalid values null with reason; a null or mixed result is a valid outcome |

## Comparison with the pilot (v1.1) — descriptive only
The pilot (24 clips, 6 positions, shadow + non-matched colours) results are reported alongside as a reference. No pooled test across the two datasets; they are different stimuli and not independent evidence.

## Required software checks before extraction
`stage_b_v2.py selftest` (held-out rows do not contribute; dual = primal; intercept; degenerate-scale stop; tie-to-orange; pooled-RMSE ≠ mean of group RMSEs; ρ zero-denominator; projection hash; frames 61–270 cannot change the encoder input). Extraction refuses without the receipt and requires the pinned vjepa2 revision, clean checkout and matching checkpoint hash.
