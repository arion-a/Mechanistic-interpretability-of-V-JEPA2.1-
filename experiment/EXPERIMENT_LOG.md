# V-JEPA 2.1 Interpretability — Experiment Log

This file's only job is to record every step, decision, bug fixed, and
experiment run on this project, in the order they happened — modeled on a
blockchain: each new action is appended as its own numbered entry ("block")
that references the one before it, and past entries are never rewritten. A
correction to an earlier step gets a new block saying so, not an edit to
history. Every block also names the key part — why that step mattered to the
overall hypothesis test, not just what was done.

This is a record, not a results report — see the
[dashboard](https://claude.ai/artifact/REykN1DyL1Zd8vuGCeJcwe) and
[methods report](https://claude.ai/artifact/R5kfsP9H3iMcCU5Yen9KFX) for that.
A mirrored, live-edited copy of this log is kept as a Claude Doc at
https://claude.ai/artifact/NbeNL1XQpCEEyX3dj8veco — this file is updated to
match after every step.

## Log

| Block | Step | Why it matters to the experiment |
|---|---|---|
| 1 | Explored repo; read the interpretability protocol (frozen V-JEPA embeddings vs. one-factor physics interventions); researched prior V-JEPA interpretability work. | Establishes the actual hypothesis under test — does V-JEPA encode each physical factor in its own compact subspace — before any data exists to test it. |
| 2 | M0: installed deps, verified `facebook/vjepa2-vitl-fpc64-256` loads (documented substitution for the docx's unpublished ViT-B/16 384px checkpoint). | Confirms the actual measuring instrument works, but flags a real deviation (different architecture/resolution) that every later result has to be read against. |
| 3 | M1: built PyBullet sphere-drop sim + TinyRenderer camera pilot. | First end-to-end sim→render skeleton, built cheaply on CPU to prove the pipeline shape before committing to GPU/Blender. |
| 4 | M2: encoded pilot clips through V-JEPA into 32,768-dim feature vectors. | First time real video became the actual object of study (z-vectors) — validates the patch/pooling extraction contract every later number depends on. |
| 5 | M3: implemented SVD / specificity / leave-one-world-out retention analysis. | This is the actual hypothesis test. Everything before this exists only to feed it data. |
| 6 | M4: built and published the first results dashboard artifact. | Made the pipeline's output visible for the first time — which is what exposed it as still too rough to trust (see blocks 7, 10-14). |
| 7 | User: dashboard was "incomprehensible" — rebuilt with what-is-this/how-is-it-computed explanations per chart, plus a separate KaTeX methods report. | Shifted the deliverable from "here are numbers" to "here is what each number means and how it was computed" — the difference between a report and evidence. |
| 8 | Connected a RunPod GPU pod via Jupyter WebSocket (sandbox blocks SSH, allows HTTPS). | Unlocks the compute the confirmatory-scale study needs; without it the project is permanently capped at CPU toy pilots. |
| 9 | Designed the parallelized pipeline: phase 1 physics-only state generator, phase 2 Blender render workers, disk-safe render→encode→delete daemon. | This architecture is what makes the real 1,200-world study *feasible at all* — without render→encode→delete, raw storage would need ~1.3TB. |
| 10 | **Bug found & fixed**: sphere only ~5px across (unresolvable) despite passing the in-frame check — bounded the speed range and recalibrated the camera. | The single most important bug in the project: if the model can't resolve the object being intervened on, every downstream number is noise, not evidence about V-JEPA's representations. |
| 11 | **Bug found & fixed**: rolling friction brought the ball to a dead stop by ~2s — removed rolling friction, shrank the speed range instead. | Same risk class as block 10: a clip where the intervention's effect goes silent for most of its length would quietly destroy signal with no error thrown. |
| 12 | **Bug found & fixed**: a "second ball" was the sphere's own harsh shadow — fixed via lower sun elevation + softer shadow angle. | A rendering artifact that could look like a second physical object — risk of contaminating the embedding with something that doesn't exist. |
| 13 | Verified free-fall trajectory against the analytic no-air-resistance equations of motion. | The ground-truth check that the simulator itself is physically correct — without it, no later result can be trusted to reflect real physics rather than a sim bug. |
| 14 | **Bug found & fixed**: bounce was physically correct but visually unreadable at a steep camera angle (~14px) — found a shallow −15° elevation giving 132px of visible bounce. | V-JEPA only ever sees pixels, not physics state — a camera angle that hides the bounce from a human hides it from the model too. |
| 15 | Saved and sent real sample clips (mp4) for the user to preview. | Let the data be checked by eye instead of taken on my word — this is what actually surfaced several of the bugs above. |
| 16 | Ran the full validated pipeline on GPU: 25 worlds × 9 clips = 225 clips, Blender EEVEE, v4 camera, corrected physics — 0 clips rejected. | The first dataset in the whole project generated entirely on the corrected pipeline — the first data actually trustworthy end-to-end. |
| 17 | **Bug found & fixed**: `weights_hash` KeyError in `svd_analysis.py` — added a fallback plus proper hash logging for future runs. | A provenance gap, not a correctness bug: without it you can't prove which exact model weights produced a given result if the checkpoint ever changes. |
| 18 | **Bug found & fixed**: BLAS thread-pool contention stalled the 225-clip SVD analysis — capped OMP/OpenBLAS/MKL threads to 4. | An engineering blocker, not a science one — but unresolved, it would make even a validated dataset unanalyzable in reasonable time at 1,200-world scale. |
| 19 | Ran the final SVD/specificity/retention analysis on all 225 clips. | **The actual scientific result of the pilot.** Gravity, restitution, and velocity show real specificity (own-basis retention 0.39–0.68, clearly the max in their row); friction shows none (0.037, the minimum in its row) — the finding the entire pipeline exists to produce. |
| 20 | Downloaded the final_pilot manifest, analysis JSON/CSV, and 2 feature vectors from the pod. | Got the result off the pod before it was shut down and became unreachable. |
| 21 | Rebuilt the dashboard and methods report against the validated 25-world data; replaced synthetic frame-crop thumbnails with real embedded mp4 clips. | Replaces every earlier, buggy-pipeline number with the first real, trustworthy account of the block-19 finding. |
| 22 | Committed and pushed the dashboard/methods-report rebuild to `claude/optimistic-mccarthy-t24gbi`. | Moves the validated result into durable, shared version control instead of only a private HTML render. |
| 23 | User shut down the GPU pod (confirmed unreachable). | Cost-control decision — but means no more raw per-clip data can be pulled after this point. |
| 24 | Archived the final_pilot manifest + 2 feature vectors to `experiment/archive/final_pilot/` (non-gitignored, with a README on what is/isn't preserved). | Preserves enough to prove provenance and regenerate the methods report's worked example; explicitly documents that 223 feature vectors + 225 state files are now unrecoverable. |
| 25 | Created the experiment log as a Claude Doc, structured as an append-only, blockchain-style record. | Makes the project's history auditable by the user directly, instead of only recoverable by re-reading chat scrollback. |
| 26 | Set up this file (`experiment/EXPERIMENT_LOG.md`) as a GitHub-committed mirror of that doc. | Makes the record durable and versioned, independent of any one platform — matching the "immutable ledger" analogy the user asked for. |
| 27 | Added a "why it matters" column to every block, per user request. | Makes the log explain each step's significance to the overall hypothesis test, not just what action was taken — so the log reads as a record of reasoning, not just of activity. |
