# Experiment log — 24-video gravity pilot

Append-only. One entry per session/decision. Record failures and deviations; never edit past entries (add a correction entry instead).

## Status board
| Stage | State |
|---|---|
| Docs received & reviewed | done (2026-10-10) |
| Environment chosen by user | this cloud sandbox (CPU, Python 3.13, ffmpeg 6.1.1, numpy 2.5.3, Pillow 12.3.0) — chosen 2026-10-10 |
| Stage A: generate / validate 24 clips, gallery | **done — awaiting review** (manifest SHA-256 `7d3baa2b…c5dd`) |
| **Dataset approval gate** | **not given** (no receipt) — waiting for user |
| Stage B: extraction, readers, report | **locked until approval** |

Inference performed: **none**. Checkpoint loaded: **no**. Readers fitted: **no**. Empirical results: **none**.

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
* **STOP — awaiting user:** inspect `dataset/gallery.html` / `contact_sheet.png` / `videos/`. To approve, the user (not the agent) runs `python3 pilot.py record-approval --manifest-sha256 7d3baa2b… --approver <name> --utc <time>` or tells the agent explicitly to record it. Any dataset edit invalidates the receipt.

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
