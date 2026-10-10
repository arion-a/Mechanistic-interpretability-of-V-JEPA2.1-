# Dataset v2.0 — confound-reduced stimuli (user-requested 2026-10-10: "also remove the known confounds")

**Status: clips generated and validated; NOT approved; no inference, no Stage B on v2.** Manifest SHA-256 `75ea34ed4ce6806bd1d065db80c76dcc6e39cc4422c0a548477813c4b0ebb6ed`. Code: `pilot_v2.py` (copy of `pilot.py` so the approved v1.1 hashes are untouched), config `pilot_config_v2.json`, output `dataset_v2/`.

## The three confounds of the v1.1 pilot, and what was done
| Confound | v1.1 | v2.0 |
|---|---|---|
| Colour not luminance-matched | orange (235,100,60) vs blue (60,140,235) | blue scaled in linear light to **(62,143,241)**; Rec.709 relative luminance orange **0.2710** vs blue **0.2702** (was 0.2710 vs 0.2571, a 5 % gap; mean-RGB gap was 10 %). Orange unchanged. |
| Ground shadow = extra position cue | hard ellipse under the ball | **removed**; validation proves frames differ from the static scene only inside the ball disc |
| Only six start positions | 6 → 24 clips, 6 folds | **12 start positions** (x0 ∈ {−60,−20,20,60} × y0 ∈ {120,140,160}) → **48 clips**, 12 folds of 44 train / 4 test |

Everything else is the v1.1 look and physics: 9 s clips, sky + striped ground, shaded sphere, bounce e = 0.6, view shifted 25 m, first 2 s only to the encoder, target at 4 s.

## Verified (24/24 checks pass, `dataset_v2/dataset_validation.json`)
Lossless FFV1 round-trip on all 48×271 frames; 271 in-order frames; ball visible and centroid within half-pixel of analytic; identical colour-pair trajectories; first impact after the 4 s target for every clip (earliest 4.84 s at g=9.8, y0=120) and within the clip; no ground penetration; rebound apex 5+e²(y0−5); luminance match; no shadow; frames 61–270 cannot change the encoder tensor.

## Not removed / new caveats
* Colour **hue and saturation** still differ (that is the variable being tested); only luminance is matched. Matching mean RGB or gamma luma is not possible simultaneously with exact relative luminance (luma′ 135.8 vs 130.0; mean RGB 131.7 vs 148.7).
* Static sky/ground are still present; the full reader will still be near-interpolating (44 training clips vs 768 features, λ = 0.01).
* y0 = 120 at g = 9.8 lands at 4.84 s — only 0.84 s after the target; the target is still bounce-free but the margin is smaller.
* More groups reduce but do not remove the "few groups" limitation; folds remain dependent. No significance claims.
* Baselines are analytically unchanged (E_CV 16.528 m, E_prior 19.6 m; groups balanced), but all counts change for analysis: n = 48, 12 folds, 24 colour pairs for J, breadth over 12 groups, storage 147,456 B vs 3,072 B payload (+49,152 B matrix).

## What is needed before Stage B on v2
1. **Your approval of these clips** (receipt against the manifest hash above; `pilot_v2.py extract` refuses without it).
2. Generalise `stage_b.py` (hard-wired to 24 clips / 6 folds) into a v2 copy, pre-specify the decision rules (unchanged from §9), and then run extraction (~48 × 5 s on CPU).
3. The lossless `.mkv` files (104 MB) are **not committed**; regenerate deterministically with `python3 pilot_v2.py generate` and verify SHA-256 against the manifest.
