# Experiment log — 24-video gravity pilot

Append-only. One entry per session/decision. Record failures and deviations; never edit past entries (add a correction entry instead).

## Status board
| Stage | State |
|---|---|
| Docs received & reviewed | done (2026-10-10) |
| Environment chosen by user | this cloud sandbox (CPU, Python 3.13, ffmpeg 6.1.1, numpy 2.5.3, Pillow 12.3.0) — chosen 2026-10-10 |
| Stage A: generate / validate 24 clips, gallery | done (dataset v1.1, manifest SHA-256 `7cef53e3…1f52`; v1.0 clips superseded) |
| **Dataset approval gate** | **approved 2026-10-10T07:56:53Z** by user, receipt `dataset/approval_receipt.json` |
| Stage B: extraction, readers, report | **done** — see `results/RESULTS.md` |

Primary prediction-retention result: **not met** (E_full 12.17 m, E_projected 22.32 m, bar 16.53 m, ρ 1.834). Colour 24/24 both. See Session 4.

## 2026-10-10 — Session 1: package setup and protocol review
* **Authorisation:** user asked to create the folder, read/understand the two supplied docs, flag inconsistencies, and prepare the environment/log. Not asked to generate videos or run inference.
* **Inputs:** `sources/MATHEMATICAL_PROTOCOL.pdf` (13 pp), `sources/CODING_AGENT_BRIEF.md`. Branch `claude/sleepy-keller-d8nuyl`.
* **Missing inputs:** `MATHEMATICAL_PROTOCOL.md`, `pilot_config.json`, `scene_plan.csv` (see I-01). Reconstructed from the PDF: `protocol/MATHEMATICAL_PROTOCOL.md`, `pilot_config.json`, `scene_plan.csv` — all labelled derived.
* **Software-only check:** `python3 tools/protocol_checks.py --write-scene-plan` — 28/28 pass (numpy 2.5.3). Confirms every hand-computable value in the protocol. Reference projection SHA-256 (numpy 2.5.3): `2a754eade0be27ac5840532226b95ad010cd849f130e1ada2459056ca1c6da2c` — informational; canonical hash is of the saved `projection.npy`.
* **Findings:** see `INCONSISTENCY_REVIEW.md` — no numeric errors; main logical issue I-02 (prior baseline can never bind: E_CV 16.528 m < E_prior 19.6 m in all six groups); missing files; FFV1 pixel-format and EMA-key gaps.
* **Not done (by design):** no video generation, no model download, no inference, no changes to existing repo files.
* **Open decisions for user:** (1) execution environment; (2) I-02 intent; (3) send original `.md`/`.json`/`.csv` if they exist; (4) approve proposed approval-receipt format (I-04).
* **Next:** after the user picks an environment — pin Pillow/numpy/ffmpeg, implement `generate` + `validate`, build gallery/contact sheet, then STOP for dataset approval.

## 2026-10-10 — Session 2: Stage A (stimulus generation) — decisions and results
* **User decisions:** execute in this cloud sandbox; keep the literal `min(E_CV, E_prior)` rule (I-02 resolved: it reduces to beating E_CV = 16.528 m; both baselines still reported separately). Experiment re-stated by user as: *does a small JEPA representation preserve future-motion information while colour becomes less accessible, at unseen start positions?* — treated as the same 24-video protocol; wording to be reported as "colour accessibility to a linear reader", not representational insensitivity.
* **Built:** `pilot.py` (`generate`, `validate`, `gallery`, `record-approval`; `extract` refuses without a matching `approval_receipt.json`; `analyze`/`report` are unimplemented Stage-B stubs).
* **Run:** `python3 pilot.py generate && python3 pilot.py validate && python3 pilot.py gallery`.
* **Outputs:** `dataset/videos/` (24 FFV1 mkv, 9.2 MB), `manifest.json`, `dataset_validation.json` (17/17 pass), `contact_sheet.png`, `gallery.html`, `viewing/*.mp4` (**derivative** H.264 for viewing only), `provenance_stageA.json`. Manifest SHA-256: `7d3baa2bb8b6d0418868d8a89d5edd313f9a41c3392b24301f6f698f8d35c5dd`. Generation is byte-deterministic (two runs, identical manifest).
* **Defects / deviations (documented):** first validation run FAILED 2 checks (timestamp tolerance 1e-6 s; centroid tolerance 0.5 px). Diagnosed as over-tight checks, not clip defects — see I-15/I-16; tolerances replaced with justified bounds and an added bias check; clips regenerated and fully re-validated. Later, the pixel format and Pillow version were written into `pilot_config.json`, changing the config hash, so clips were regenerated again; the manifest hash above is from the final run.
* **Not done:** no checkpoint download, no model load, no embeddings, no readers, no results.
* **STOP — awaiting user:** inspect `dataset/gallery.html` / `contact_sheet.png` / `videos/`. (superseded by Session 3) To approve, the user (not the agent) runs `python3 pilot.py record-approval --manifest-sha256 7d3baa2b… --approver <name> --utc <time>` or tells the agent explicitly to record it. Any dataset edit invalidates the receipt.

## 2026-10-10 — Session 3: dataset v1.1 (user-requested restyle + bounce)
* **Trigger:** user reviewed v1.0 clips: balls never touch the ground, no bounce, dark flat background. Asked for normal-looking video: colour, 3D feel, normal physics.
* **Decisions (user):** shaded scene with the same side camera; longer clips so the bounce happens after the 4 s target; user asked whether length changes the calculations — answered no (encoder frames, target, baselines unchanged), then "go ahead". Restitution 0.6 and 9 s were my stated defaults.
* **Changes:** see `protocol/DATASET_V1_1_CHANGES.md` (ground, sky, shaded ball, shadow, 271 frames, bounce e=0.6, view shifted 25 m so ground is visible).
* **Validation:** `dataset_validation.json` 22/22 pass, incl. new checks: first impact after 4 s and within the clip, no ground penetration, rebound present, apex = 5+e²(y0−5) = 53.6 m, first 5 s identical to v1.0 formula, frames 61+ do not affect encoder tensor.
* **Process notes:** v1.0 manifest `7d3baa2b…c5dd` is superseded. New manifest SHA-256 `7cef53e34ace1c1147e92b79be750efde10d24e9ae794b815d93ec20e4641f52`. Lossless videos are now 53 MB total (was 9 MB). Generation remains deterministic.
* **Not done:** no model, no embeddings, no readers, no results. No approval receipt.
* **STOP — awaiting user:** inspect clips; approval must reference the new hash.

## 2026-10-10 — Session 4: dataset approval and Stage B
* **Approval:** user instruction "approve the clips and start stage B". Receipt written against manifest `7cef53e3…1f52` at 07:56:53Z (`dataset/approval_receipt.json`). `extract` refused before it existed.
* **Environment:** this cloud sandbox, CPU only (4 cores), Python 3.13, torch 2.14.1+cpu, torchvision 0.29.1+cpu, timm 1.0.30, numpy 2.5.3, Pillow 12.3.0, matplotlib 3.11.2. vjepa2 cloned and checked out at `204698b…` (clean). Checkpoint sha256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.
* **Before inference:** `stage_b.py selftest` 10/10 passed (held-out rows do not affect scale/mean/weights, dual=primal, intercept, degenerate-scale stop, tie-to-orange, pooled-RMSE, rho zero-denominator, projection reproducibility, frames 61-270 cannot change the input tensor).
* **Run:** `stage_b.py extract` (24 forward passes, ~5 s each, float32 CPU; 16 frames, tensor [1,3,16,384,384]), then `analyze`, then `report`. Projection sha256 `2a754eade0be27ac5840532226b95ad010cd849f130e1ada2459056ca1c6da2c` (same as the pre-run reference).
* **Result:** see `results/RESULTS.md`. Primary retention rule not met; colour 24/24 for both representations; no selective colour loss.
* **Deviations / defects (documented):**
  1. A first extraction launch was aborted by me after ~14 clips because I edited `stage_b.py` while it ran (provenance code hash would not have matched); its partial cache was discarded. No outputs from it were used.
  2. After the first full run I corrected cosmetic plot-layout defects (clipped footer, label overlaps) in `report_plots.py`; because that file is in the code hash, the whole pipeline was re-run (twice). Embeddings, predictions and summary were verified **bit-identical** across all complete runs. No seed, penalty, layer or data choice changed.
  3. A cleanup command was blocked by a safety check (relative glob from the wrong directory); no files were removed by it. Reruns simply overwrite outputs.
  4. `provenance.json` field `git_sha` is the commit at run time (parent of the commit holding the final `report_plots.py` edit); `code_sha256` is authoritative and equals the SHA of the committed code files.
* **Not done:** no new seeds/penalties/layers/dims/gravities/scenes, no significance claims, no further experiments (per protocol §10.8).

## Entry template
```
## YYYY-MM-DD — Session N: <title>
* Authorisation / stage:
* Environment (host, GPU, python, package versions, git SHA, code hash, config hash):
* Actions & commands:
* Outputs (paths, SHA-256):
* Deviations / defects (documented; affected outputs rerun?):
* Open decisions:
```
