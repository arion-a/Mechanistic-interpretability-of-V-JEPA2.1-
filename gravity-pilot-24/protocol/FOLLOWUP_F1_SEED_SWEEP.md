# Follow-up F1 — projection-seed robustness (pre-specified before running; user-requested 2026-10-10)

**Status: exploratory follow-up. It does not replace or re-score the pilot's primary result** (seed 20261009: E_projected 22.32 m, ρ 1.83, primary rule not met). That result stands as reported.

## Question
Is the failure of the 16-D random projection specific to the one seed (20261009), or typical of random 16-D projections of these embeddings?

## Fixed in advance (nothing here may change after seeing results)
* **Inputs:** the existing 24 frozen embeddings (`results/embeddings_full.npy`) from the approved dataset v1.1. **No new videos, no new model inference.**
* **Seeds:** the original 20261009 plus 100 new seeds, the consecutive integers **20261010 … 20261109**. The list was chosen here, before any computation, not from results.
* **Projection:** identical recipe to the pilot (PCG64 → 16×768 ±1 entries → (2b−1)/4, float32). Dimension stays **16**. No PCA, no other dimensions.
* **Readers, folds, scaling, λ = 0.01, metrics, baselines:** exactly as in protocol v1.0 §5–9, via the same `readers.py`. E_full is the pilot's 12.17 m (unchanged, projection-independent).
* **Per seed:** E_projected, ρ = E_proj/E_full, six group RMSEs, number of groups beating both baselines, colour correct (of 24), ΔA, J_projected, primary-rule pass/fail (E_proj < min(E_CV,E_prior) **and** ρ ≤ 1.10).
* **Check:** the row for seed 20261009 must reproduce the pilot's numbers exactly; otherwise the sweep is invalid.

## Reporting rules
* Report **all 101 seeds**; no seed is dropped, ranked as "the best", or substituted for the original.
* Summaries: fraction of seeds passing the primary rule; min / 25 % / median / 75 % / max of E_proj and ρ; colour-correct distribution; fraction with E_proj below the bar.
* Descriptive only: **no p-values, no confidence intervals, no significance claim.** The seeds are not independent trials of anything beyond the random matrix; the 24 clips and six groups are unchanged, so this says nothing new about generalisation to other clips.
* Pre-stated reading: few passes ⇒ the original failure is typical of random 16-D projections here; many passes ⇒ the original draw was unlucky. A "best seed" is never a result.
