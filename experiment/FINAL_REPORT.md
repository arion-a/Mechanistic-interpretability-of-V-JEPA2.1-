# V-JEPA 2.1 Physics-Intervention Geometry — Final Report

*Mirrored from the [Claude Doc](https://claude.ai/artifact/7b5a0e8e-5302-4c78-acd8-75b9b2ffa190). Kept in sync manually, not auto-synced.*

## Executive Summary

This project tests whether V-JEPA 2's frozen video embeddings encode distinct physical factors — gravity, restitution, friction, initial velocity — as separable, compact subspaces, following the design in `01_3D_VJEPA_Research_Agenda.docx`. A fully deterministic PyBullet + Blender EEVEE pipeline renders a sphere drop/bounce scene in 9 versions per world (1 baseline + 8 single-factor interventions), encodes every clip through the frozen `facebook/vjepa2-vitl-fpc64-256` model, and analyzes the resulting embedding differences with uncentered SVD, leave-one-world-out retention, and cross-factor specificity.

A 25-world pilot validated the pipeline end-to-end first, catching and fixing four real defects (an unresolvably small sphere, motion that froze, a lighting artifact, an unreadable camera angle). The confirmatory study then ran at full design-doc scale: **1,200 worlds, 10,797 accepted clips**.

**Headline finding:** three of four factors — gravity, restitution, and velocity — show a clear, factor-specific subspace: a basis fit on that factor's own effect predicts its held-out variation better than any other factor's basis does (retention 0.667 / 0.727 / 0.485, against a 0.000244 chance floor). Friction is the consistent exception at both pilot and confirmatory scale: the model does respond to it, but the response does not organize into a compact, factor-specific direction (retention 0.096, barely above chance). All three specific factors' retention scores *increased* from pilot to confirmatory scale (48x more data) rather than regressing toward chance — evidence this is real structure, not a small-sample artifact.

## Hypothesis & Background

V-JEPA 2 is trained purely on video prediction in latent space, with no explicit physics supervision. If it has nonetheless learned an internal model of physical dynamics, that structure should be visible in its frozen embeddings: changing one physical constant (gravity, restitution, friction, or an object's initial velocity) while holding everything else identical should shift the embedding in a direction specific to that constant, separable from the directions other constants shift it in.

The **null hypothesis** is that V-JEPA's embedding space mixes these factors together, or encodes some of them only diffusely (e.g. as generic "looks different" noise) rather than as a factor-specific direction any basis could isolate.

This follows the design specified in `01_3D_VJEPA_Research_Agenda.docx` and `02_3D_VJEPA_Mathematical_Framework.docx`: a controlled single-factor intervention design (one physics constant changed per clip, all else identical and deterministic), a fixed encoder architecture, and an uncentered-SVD-based specificity/retention test. One documented deviation from the design docs: the docs specify an unpublished ViT-B/16 384px checkpoint; this project substitutes the published `facebook/vjepa2-vitl-fpc64-256` (ViT-L/16, 256px, 64-frame) checkpoint, a larger and differently-shaped model. Every result below should be read against that substitution (see Limitations).

## Experimental Pipeline

Four sequential stages, run per world:

1. **Simulate.** PyBullet samples one random anchor state (position, heading, speed, vertical velocity) per world, deterministically seeded (SHA-256 of root seed 42 + world index — never `hash()`, PID, or clock). From that one anchor, physics is simulated 9 times: once at baseline settings, then once for each of 8 single-factor interventions (2 doses x 4 factors: gravity, restitution, friction, velocity). Only the changed constant differs between any two of a world's 9 runs.
2. **Render.** Blender 4.2.3 LTS (EEVEE), headless, GPU-accelerated, renders each of the 9 runs into a 180-frame (6s @ 30fps) 256x256 clip from a fixed camera. A visibility gate rejects any clip where the sphere leaves the frustum (3 rejected out of 10,800 at confirmatory scale).
3. **Encode.** 64 frames are uniformly sampled across each clip and passed through the frozen `facebook/vjepa2-vitl-fpc64-256` encoder. Its patch/tubelet output is spatially pooled per time-step and concatenated across all 32 time-steps into one 32,768-dimensional vector z per clip (see Mathematical Framework).
4. **Analyze.** For each world and factor, z(intervention) - z(baseline) = Δz isolates that factor's effect on the representation. Δz vectors are stacked per factor and analyzed with uncentered SVD, leave-one-world-out retention, and cross-factor specificity.

Render and encode run concurrently on the GPU pod as a disk-safe render→encode→delete pipeline: each clip is deleted immediately after encoding, keeping disk usage flat instead of needing ~1.3TB for the full confirmatory dataset's raw frames.

## Mathematical Framework

**Physical state.** A 13-number state per physics step: 3D position, a 4-number orientation quaternion, 3D linear velocity, 3D angular velocity.

```
S_t = (p_t, q_t, v_t, ω_t) ∈ R^13
```

**Encoding one clip into one vector z.** The encoder's 3D patch embedding divides the 64x256x256 input into 32 time-tubelets x 256 spatial patches, each a 1,024-wide vector; spatial patches are averaged within each time-step and the 32 resulting vectors concatenated:

```
z = ⊕(t=1..32) [ (1/256) Σ(s=1..256) H_{t,s,:} ]  ∈  R^32768
```

**Δz: the effect of one intervention.** Both clips share the identical anchor, seed, appearance and camera — only one physics constant differs — so Δz isolates that constant's effect:

```
Δz = z(intervention clip) − z(baseline clip)  ∈  R^32768
```

**Uncentered SVD — the compactness spectrum.** Stack every Δz for one factor as a row of M_f, then factor it into orthogonal directions ranked by explained energy:

```
M_f = U Σ V^T
r90 = min{ r : Σ(i=1..r) σ_i² / Σ(i=1..n_f) σ_i² ≥ 0.90 }
```

**Mean-shift energy fraction.** How much of a factor's average Δz length is one single consistent shift versus varying per world:

```
m_f = (1/n_f) Σ Δz_i
fraction = ‖m_f‖² / [ (1/n_f) Σ ‖Δz_i‖² ]
```

**Leave-one-world-out (LOWO) retention.** Whether a rank-r basis learned from some worlds predicts a held-out world's direction of change:

```
V_{-w} = top-r right singular vectors of M_f fit on all rows except world w
retention_i = 1 − ‖Δz_i − Δz_i V_{-w}^T V_{-w}‖² / ‖Δz_i‖²
```

**Cross-factor specificity.** The same formula, but the basis is fit on one factor and tested on another's held-out Δz — the diagonal reuses LOWO's held-out retention so every cell, diagonal included, is out-of-sample:

```
specificity[f_row, f_col] = mean_i( 1 − ‖Δz_i − Δz_i V_{f_row}^T V_{f_row}‖² / ‖Δz_i‖² ),  Δz_i ∈ M_{f_col}
```

**Chance level.** By symmetry, a random rank-r subspace in a d-dimensional space captures r/d of a fixed vector's energy in expectation, giving the floor every retention number is judged against:

```
E[retention_random] = r/d = 8/32768 ≈ 0.000244
```

**Computational note.** At confirmatory scale, LOWO needs ~4,800 folds (1,200 worlds x 4 factors); a full SVD per fold would cost ~71 hours. Each fold only ever uses the top-8 right singular vectors, so a randomized/truncated SVD (`sklearn.utils.extmath.randomized_svd`) computes the identical target subspace — not an approximation of a different quantity — in ~1.6 hours total.

## Process Summary

Condensed from the full 54-block append-only experiment log (`experiment/EXPERIMENT_LOG.md`), chronological.

| Phase | Key event | Outcome |
| --- | --- | --- |
| Setup | Explored repo, read the interpretability protocol, installed deps, verified the encoder loads. | Established the hypothesis and confirmed the measuring instrument works. |
| Pilot build | Built PyBullet sim, TinyRenderer pilot, V-JEPA encoding, SVD/specificity/retention analysis, first dashboard. | First end-to-end skeleton — immediately exposed as too rough to trust. |
| Bug-fixing | Found and fixed 4 real defects: unresolvably small sphere, motion freezing at ~2s, a shadow artifact mistaken for a second ball, and a camera angle hiding the bounce. | Each bug would have silently corrupted every downstream number — the model only ever sees pixels, not physics state. |
| Pilot run | Ran the fully validated pipeline on GPU: 25 worlds x 9 clips, Blender EEVEE, 0 clips rejected. | First dataset generated entirely on the corrected pipeline — first trustworthy result (retention 0.037-0.678). |
| Infra setup | Provisioned 3 successive RunPod GPU pods; solved persistent-volume storage; benchmarked real per-clip render/encode timing (not a guess) before committing to cost. | Made the ~48x-larger confirmatory run technically and financially predictable ($6-$8 total). |
| Confirmatory launch | Phase 1 generated all 1,200 worlds (10,800 clips) in 12.85 min; Phase 2 launched 4 concurrent render shards + an encode-and-delete daemon. | The actual confirmatory data generation run begins. |
| Disk-backlog incident | Encode fell behind render under real 4-way GPU contention (a mode the pre-run benchmark never tested); paused 2 of 4 shards, then corrected a flawed pause plan that would have permanently dropped ~32% of clips, running the paused shards as a second wave instead. | Caught before any clips were lost; avoided both an out-of-disk crash and a silent, undetected missing-data bug. |
| Wave 1 & 2 completion | Both waves of rendering completed cleanly; encode stayed fully caught up throughout. | No repeat of the disk-safety incident across ~9 hours of sustained runtime. |
| Pod crash & recovery | The pod went fully unreachable at ~88% complete. A new pod was deployed onto the same persistent volume; an exact byte-count and SHA-256 baseline confirmed zero real data loss (an apparent mismatch was a transient volume-mount sync delay). The exact 1,110 missing clips were recomputed from disk truth and re-rendered. | Full recovery from a real infrastructure failure with zero data loss, verified by checksum rather than assumed. |
| Render+encode complete | All 10,797 accepted clips verified present against the manifest. | The full confirmatory dataset is complete. |
| SVD analysis at scale | A full SVD per leave-one-world-out fold was benchmarked at ~71 hours (infeasible) before running blind; switched to randomized SVD for the exact top-8 subspace, cutting this to ~1.6 hours. | Produced the real confirmatory-scale specificity and retention numbers below. |
| Reporting | Dashboard and methods report rebuilt with the real N=1,200 numbers and republished; this final report written. | Closes out the study's deliverables. |

## Results

**Run:** 1,200 base worlds, 10,797 accepted clips (10,800 rendered, 3 rejected — 0.03%), feature dimension 32,768, rank 8, model `facebook/vjepa2-vitl-fpc64-256`, weights hash `712c235142eba2e1` (confirmed identical across the pre-crash and post-recovery encoding passes).

**Per-factor spectrum, mean-shift, and retention:**

| Factor | Pairs (n) | r90 (of ~2,400) | Mean-shift energy | ‖mean Δz‖ | LOWO retention (1,200 folds) |
| --- | --- | --- | --- | --- | --- |
| Gravity | 2,400 | 105 | 3.50% | 7.906 | 0.667 |
| Restitution | 2,400 | 46 (most compact) | 20.33% | 28.074 | 0.727 |
| Friction | 2,400 | 142 (least compact) | 0.05% | 0.060 | 0.096 |
| Velocity | 2,397 | 133 | 2.60% | 8.785 | 0.485 |

Chance-level retention for a rank-8 subspace in 32,768 dimensions: 0.000244. Every factor sits far above this floor — gravity, restitution and velocity by three orders of magnitude, friction by roughly two.

**Cross-factor specificity matrix** (rows = fitted basis, columns = tested factor; every cell, including the diagonal, is a held-out score):

| Fitted on \ Tested on | Gravity | Restitution | Friction | Velocity |
| --- | --- | --- | --- | --- |
| Gravity | **0.667** | 0.331 | 0.026 | 0.042 |
| Restitution | 0.277 | **0.727** | 0.022 | 0.053 |
| Friction | 0.067 | 0.055 | 0.096 | 0.131 |
| Velocity | 0.046 | 0.054 | 0.074 | **0.485** |

Gravity's, restitution's, and velocity's own-basis diagonal is more than 2x its row's next-best cell. Friction's diagonal (0.096) is beaten by velocity's basis (0.131) — the one row where the diagonal is not the row maximum.

**Pilot-to-confirmatory comparison** (25 worlds -> 1,200 worlds, 48x more data):

| Factor | Pilot retention (n≈25) | Confirmatory retention (n=1,200) |
| --- | --- | --- |
| Gravity | 0.638 | 0.667 |
| Restitution | 0.678 | 0.727 |
| Friction | 0.037 | 0.096 |
| Velocity | 0.391 | 0.485 |

Every factor's retention *increased* with scale, including friction's (still near chance, but no longer indistinguishable from it). At pilot scale, r90 for every factor sat almost at the sample count itself (~8-10 out of ~40-50 pairs) — a sample-size ceiling rather than a real compactness signal; at confirmatory scale, r90 is far below the sample count and the ordering it reveals (restitution most compact, friction least) is the reverse of what the noisy pilot-scale estimate suggested.

## Interpretation

**The core hypothesis holds for three of four factors.** Gravity, restitution, and velocity each have a basis that beats every other factor's basis at predicting their own held-out variation — the signature of a real, factor-specific subspace, not shared noise. That this held up, and strengthened, at 48x the pilot's scale rules out the main alternative explanation (a small-sample artifact that would wash out or reverse with more data).

**Friction is a genuine negative result, not a measurement failure.** Its mean-shift energy fraction (0.05%) is two orders of magnitude below the other three factors, meaning friction's effect on Δz is almost entirely per-world noise rather than a consistent shift — and its retention, while statistically above chance, is the one factor whose own diagonal is beaten by another factor's basis. Two readings are both consistent with the data: either V-JEPA's embedding is less sensitive to friction as a distinct physical quantity in this scene (a 6-second single-bounce clip may simply not surface much friction-driven visual difference to encode), or friction's effect rides on the same visual cues (contact geometry, rolling behavior) that gravity and restitution also produce, diluting its own signature rather than eliminating it.

**Restitution has the strongest, most efficient signature** (highest retention, lowest r90, though also by far the largest mean-shift energy fraction — restitution changes the bounce height dramatically and consistently, which is intuitive: it is the single constant most directly and visibly tied to what a viewer sees a bouncing ball do).

## Related Work & Theoretical Framing

**Direct comparisons — physics interpretability in video world models.** [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) is the closest methodological relative: layerwise probing, subspace geometry, and attention ablations on video transformer encoders, studying motion variables (speed, acceleration, direction). It finds a "Physics Emergence Zone" at intermediate depth where physics becomes linearly decodable, but organized as *distributed, geometrically structured* representations rather than clean, physics-engine-like factorization — direction, for instance, has circular multi-feature structure, not one dedicated axis. This study finds a notably cleaner separation for 3 of 4 factors than that degree of entanglement would predict, likely because differencing two otherwise-identical clips (Δz) cancels shared scene/appearance structure and isolates only the intervention's effect — a much cleaner signal than probing one clip's absolute-quantity encoding.

[How Do Video Foundation Models Encode Intuitive Physics?](https://arxiv.org/abs/2606.09646) compares V-JEPA, VideoMAE, and a diffusion model on physics-probing benchmarks (MVP, IntPhys2) and finds V-JEPA strongest, especially with probes that model temporal dynamics — consistent with this study's choice to keep the encoder's 32-step temporal resolution before differencing rather than collapsing to one pooled vector.

Meta's own violation-of-expectation results ([Intuitive physics understanding emerges from self-supervised pretraining](https://www.alphaxiv.org/abs/2502.11831)) show V-JEPA scoring >95% on IntPhys by measuring prediction-error "surprise" at physically impossible events — a behavioral, black-box test. This study's specificity/retention test is a complementary, representational, white-box question: not just whether the model *behaves* as if it understands a factor, but whether that behavior is backed by a legible, factor-specific direction in its embedding versus a more entangled computation.

**Precedent for factor disentanglement, and its limits.** Older supervised work built models with explicit disentangled dimension blocks for mass/speed/friction and found mass and friction could not be cleanly separated, because friction's visual signature (how quickly motion decays) is confounded with mass's signature in the same rolling-motion cues. This study finds an analogous confound one factor over: friction's own basis (0.096) is beaten by velocity's basis (0.131) at predicting friction's held-out variation — friction's effect in a short single-bounce clip appears to ride on the same kinematic cues velocity produces, rather than carving out its own direction.

**LLM interpretability parallel.** This study's core operation — difference two matched conditions (Δz), fit a subspace to many such differences for one factor, and test whether it predicts held-out differences — is structurally the same move used throughout the LLM "linear representation hypothesis" literature. Representation engineering (Zou et al., 2023) extracts a "concept direction" from paired contrastive activations the same way; function vectors and task vectors (Todd et al. 2023; Hendel et al. 2023) extract one vector from many task demonstrations and show it generalizes to new inputs, directly analogous to this study's leave-one-world-out generalization test. The specificity matrix here — does factor A's basis explain factor B's variation — mirrors how LLM steering-vector work checks whether a concept vector (e.g. sentiment) also moves unrelated behavior, as evidence for or against a vector being causally specific rather than a shared "generic difference" direction.

**World-model probing in sequence models — the Othello-GPT lesson.** The most directly comparable "does a model represent a hidden environment variable in a legible subspace" result outside video work is [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941) (Nanda, Lee & Wattenberg, 2023), building on Li et al.'s Othello-GPT. The original work found a transformer trained only on move sequences builds an internal model of the board, but recoverable only with a *nonlinear* probe when board state is coded as absolute "black/white" occupancy. Nanda et al. showed a clean *linear* subspace exists after all, once the probe is re-expressed in the model's own natural coordinate system ("my colour" vs "opponent's colour") rather than fixed colour identity. The lesson for this study: a negative specificity result (friction) is a claim about *this* basis — uncentered SVD on Δz under baseline-minus-intervention differencing — not proof that no linear subspace under any better-aligned basis could describe friction's effect. This is a genuine limitation worth stating alongside the ones above: the friction null result is basis-dependent by construction, the same way Othello-GPT's board state briefly looked nonlinear only because of an arbitrary coordinate choice.

[Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207) (Gurnee & Tegmark, 2023) is the LLM-world-model analogue of this study's whole premise: linear probes on LLM activations recover geographic and temporal coordinates the model was never explicitly supervised on — evidence that this general shape of finding (frozen embeddings encoding external-world variables as linear directions) recurs across very different architectures and modalities, not something special to video encoders.

[Distributed Alignment Search](https://arxiv.org/abs/2303.02536) (Geiger et al., 2023) formalizes almost exactly this study's core question as a general method: given a hypothesized causal variable, does a linear subspace of a model's representation align with it — tested causally (swap the subspace's value between two inputs, check whether downstream behavior changes as the causal variable predicts) rather than only correlationally. This study's retention and specificity metrics are correlational (how well a subspace's projection predicts Δz); a DAS-style intervention — substituting one world's fitted gravity-subspace component into another world's embedding and checking whether a downstream probe now reads out the swapped value — would be a natural, causally stronger follow-up test of the same finding.

**Why friction might be the exception: superposition.** [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) (Elhage et al., Anthropic, 2022) gives a concrete account of when a feature earns its own dedicated direction versus being folded into superposition with others: it depends on the feature's *importance* to the training objective and its *sparsity*, relative to available representational capacity. If a 6-second single-bounce clip's rendered appearance is only weakly and inconsistently sensitive to the friction coefficient — unlike gravity, restitution, or velocity, which visibly reshape the trajectory every time — friction would be exactly the kind of low-importance, diffuse feature superposition theory predicts gets no dedicated direction. This reframes the friction result as a specific, falsifiable prediction of an established theory, not only a null finding of this particular pipeline.

**What this study adds.** Prior V-JEPA physics work either probes single absolute quantities from raw activations at a chosen layer, or tests behavioral surprise on curated possible/impossible pairs. This study instead applies the LLM interpretability toolkit — matched-pair differencing, subspace fitting, cross-condition specificity, leave-one-out retention against an analytic chance baseline — to a video world model's embeddings, at design-doc scale (1,200 worlds, fully controlled and deterministic). To our knowledge this is a novel application of that linear-subspace specificity methodology to factor-of-variation disentanglement in a video world model at this scale.

## Limitations & Deviations from the Design Docs

- **Model substitution.** The design docs specify an unpublished ViT-B/16 384px checkpoint; this study uses the published `facebook/vjepa2-vitl-fpc64-256` (ViT-L/16, 256px, 64-frame fixed clip length). A larger model at lower resolution is a real architectural difference — results characterize this specific checkpoint, not "V-JEPA 2" in general.
- **Single scene type.** Every clip is a sphere drop/bounce on a flat plane. The findings say nothing about whether the same specificity pattern holds for more complex scenes, multiple objects, or non-rigid-body physics.
- **Weights-hash logging gap (pilot only).** The 25-world pilot's encode daemon did not log a weights hash, so that run's `weights_hash` field is `not_recorded_by_sweeper`; this was fixed before the confirmatory run, whose weights hash (`712c235142eba2e1`) is confirmed identical across the pre-crash and post-recovery encoding passes.
- **Friction and velocity's slightly unequal pair counts.** Gravity and restitution have exactly 2,400 pairs (2 x 1,200 worlds); velocity has 2,397 (3 pairs lost to the visibility gate, consistent with the 3 total rejected clips).
- **A real infrastructure interruption mid-run.** The confirmatory dataset was completed across a pod crash and recovery (see Process Summary); recovery was verified by exact file-count and SHA-256 manifest checksums against a pre-crash baseline, not assumed. No evidence of any corrupted or duplicated clip was found, but this is a real deviation from an uninterrupted run worth stating plainly.
- **Rank fixed at 8 throughout**, matching the design docs' specified confirmatory rank exactly; this was not re-tuned or searched over at confirmatory scale.
- **The friction null result is basis-dependent.** As the Othello-GPT precedent shows (Related Work), a factor appearing not to have a compact linear subspace under one basis (here, uncentered SVD on baseline-minus-intervention Δz) does not rule out a differently-aligned linear basis finding one. This study did not search over alternative bases for friction specifically.
- **Raw per-clip feature vectors are not archived in this repository.** Only the computed analysis outputs (`pilot_analysis.json`, `specificity_matrix.csv`) and 2 sample feature vectors are committed; the full 10,797-vector raw dataset lives only on the GPU pod's persistent network volume.

## Conclusion & Future Work

At full design-doc scale (1,200 worlds, 10,797 clips), V-JEPA 2's frozen embeddings encode gravity, restitution, and initial velocity as compact, factor-specific subspaces — the effect strengthened rather than weakened with 48x more data, the signature of real structure. Friction does not show the same specific structure; the model responds to it, but not through a direction its own basis can isolate better than another factor's basis can.

**Future work:**
- Re-run with the design docs' original ViT-B/16 384px checkpoint if it becomes available, to isolate whether the friction result is a property of V-JEPA representations generally or specific to this checkpoint.
- Test whether friction's weak specificity persists in scenes where it has a more visually dominant role (e.g. a rolling object over a long distance, rather than a single bounce).
- Preserve or re-verify the full raw feature-vector dataset (currently only on the pod's persistent volume) if further re-analysis at the raw-vector level is planned.
- Per the user's standing instruction recorded during this session: use GPU acceleration for future CPU-bound analysis steps (e.g. a CUDA-based randomized SVD) where it would meaningfully help, rather than only for the neural-network forward pass.

## Links & Provenance

| What | Where |
| --- | --- |
| Results dashboard (interactive, charts) | [Intervention Geometry — Confirmatory Study](https://claude.ai/artifact/REykN1DyL1Zd8vuGCeJcwe) |
| Methods report (math worked through with real numbers) | [Intervention Geometry Methods](https://claude.ai/artifact/R5kfsP9H3iMcCU5Yen9KFX) |
| Final report (this document, as a live Claude Doc) | [Final Report doc](https://claude.ai/artifact/7b5a0e8e-5302-4c78-acd8-75b9b2ffa190) |
| Full append-only experiment log (54 blocks, blockchain-style) | [Experiment Log doc](https://claude.ai/artifact/aee63a8d-152a-46a2-9a21-a138243eb048) · mirrored at `experiment/EXPERIMENT_LOG.md` |
| Source repository | `arion-a/Mechanistic-interpretability-of-V-JEPA2.1-`, branch `claude/optimistic-mccarthy-t24gbi` |
| Analysis outputs (committed) | `experiment/results/confirmatory/pilot_analysis.json`, `specificity_matrix.csv` |
| Analysis code | `experiment/analysis/svd_analysis.py` |
| Pipeline code | `experiment/sim/`, `experiment/remote/blender_render_worker.py`, `experiment/remote/encode_and_sweep.py` |
| Design specification | `01_3D_VJEPA_Research_Agenda.docx`, `02_3D_VJEPA_Mathematical_Framework.docx` |

All numbers in this report are read directly from `pilot_analysis.json` (confirmatory run, N=1,200 worlds, 10,797 clips) or the pilot's own recorded results (block 19 / block 21 of the experiment log), not illustrative.
