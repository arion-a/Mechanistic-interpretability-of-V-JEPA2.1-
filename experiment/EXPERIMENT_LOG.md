# V-JEPA 2.1 Interpretability — Experiment Log

This file's only job is to record every step, decision, bug fixed, and
experiment run on this project, in the order they happened — modeled on a
blockchain: each new action is appended as its own numbered entry ("block")
that references the one before it, and past entries are never rewritten. A
correction to an earlier step gets a new block saying so, not an edit to
history.

This is a record, not a results report — see the
[dashboard](https://claude.ai/artifact/REykN1DyL1Zd8vuGCeJcwe) and
[methods report](https://claude.ai/artifact/R5kfsP9H3iMcCU5Yen9KFX) for that.
A mirrored, live-edited copy of this log is kept as a Claude Doc at
https://claude.ai/artifact/NbeNL1XQpCEEyX3dj8veco — this file is updated to
match after every step.

## Log

| Block | Step |
|---|---|
| 1 | Explored repo; read the interpretability protocol (frozen V-JEPA embeddings vs. one-factor physics interventions); researched prior V-JEPA interpretability work. |
| 2 | M0: installed deps, verified `facebook/vjepa2-vitl-fpc64-256` loads (documented substitution for the docx's unpublished ViT-B/16 384px checkpoint). |
| 3 | M1: built PyBullet sphere-drop sim + TinyRenderer camera pilot. |
| 4 | M2: encoded pilot clips through V-JEPA into 32,768-dim feature vectors. |
| 5 | M3: implemented SVD / specificity / leave-one-world-out retention analysis. |
| 6 | M4: built and published the first results dashboard artifact. |
| 7 | User: dashboard was "incomprehensible" — rebuilt with what-is-this/how-is-it-computed explanations per chart, plus a separate KaTeX methods report. |
| 8 | Connected a RunPod GPU pod via Jupyter WebSocket (sandbox blocks SSH, allows HTTPS). |
| 9 | Designed the parallelized pipeline: phase 1 physics-only state generator, phase 2 Blender render workers, disk-safe render→encode→delete daemon (avoids ~1.3TB storage for the full study). |
| 10 | **Bug found & fixed**: sphere only ~5px across (unresolvable) despite passing the in-frame check — bounded the speed range and recalibrated the camera. |
| 11 | **Bug found & fixed**: rolling friction brought the ball to a dead stop by ~2s ("nothing happens after 2s") — removed rolling friction, shrank the speed range instead. |
| 12 | **Bug found & fixed**: a "second ball" was the sphere's own harsh shadow — fixed via lower sun elevation + softer shadow angle. |
| 13 | Verified free-fall trajectory against the analytic no-air-resistance equations of motion. |
| 14 | **Bug found & fixed**: bounce was physically correct but visually unreadable at a steep camera angle (~14px of motion) — found a shallow −15° elevation giving 132px of visible bounce. |
| 15 | Saved and sent real sample clips (mp4) for the user to preview. |
| 16 | Ran the full validated pipeline on GPU: 25 worlds × 9 clips = 225 clips, Blender EEVEE, v4 camera, corrected physics — 0 clips rejected. |
| 17 | **Bug found & fixed**: `weights_hash` KeyError in `svd_analysis.py` (sweeper never recorded it) — added a fallback plus proper hash logging for future runs. |
| 18 | **Bug found & fixed**: BLAS thread-pool contention stalled the 225-clip SVD analysis — capped OMP/OpenBLAS/MKL threads to 4. |
| 19 | Ran the final SVD/specificity/retention analysis on all 225 clips. Result: gravity, restitution, and velocity show real specificity (own-basis retention 0.39–0.68); friction shows none (0.037, the weakest cell in its row). |
| 20 | Downloaded the final_pilot manifest, analysis JSON/CSV, and 2 feature vectors from the pod. |
| 21 | Rebuilt the dashboard and methods report against the validated 25-world data; replaced synthetic frame-crop thumbnails with real embedded mp4 clips; republished both artifacts in place. |
| 22 | Committed and pushed the dashboard/methods-report rebuild to `claude/optimistic-mccarthy-t24gbi`. |
| 23 | User shut down the GPU pod (confirmed unreachable). |
| 24 | Archived the final_pilot manifest + 2 feature vectors to `experiment/archive/final_pilot/` (non-gitignored, with a README on exactly what is/isn't preserved); committed and pushed. |
| 25 | Created the experiment log as a Claude Doc, structured as an append-only, blockchain-style record per the user's request. |
| 26 | Set up this file (`experiment/EXPERIMENT_LOG.md`) as a GitHub-committed mirror of that doc, to be updated after every future step. |
