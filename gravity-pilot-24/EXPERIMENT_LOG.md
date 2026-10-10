# Experiment log — 24-video gravity pilot

Append-only. One entry per session/decision. Record failures and deviations; never edit past entries (add a correction entry instead).

## Status board
| Stage | State |
|---|---|
| Docs received & reviewed | done (2026-10-10) |
| Environment chosen by user | **pending — user decision** |
| Stage A: generate / validate 24 clips, gallery | not started |
| **Dataset approval gate** | **not given** (no receipt) |
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
