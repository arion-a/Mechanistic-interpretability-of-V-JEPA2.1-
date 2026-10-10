# 24-video gravity pilot (V-JEPA 2.1, frozen features)

**Protocol v1.0 · 9 Oct 2026 · dataset v1.1 approved · Stage B complete (results in `results/RESULTS.md`): full embedding beats baselines, 16-D projection does not retain position accuracy (ρ = 1.83), colour 24/24 both.**

A small, pre-specified test of whether frozen V-JEPA 2.1 ViT-B features from the **first 2 s** of a 5 s coloured-ball video let a linear ridge reader predict the ball's position at **4 s**, whether a fixed seeded **16-D random projection** keeps that accuracy within 10 %, and whether orange-vs-blue colour stays readable. 24 clips = 6 start positions × 2 gravities (9.8, 4.9) × 2 colours; 6 leave-one-position-out folds.

| File | Purpose |
|---|---|
| [`INCONSISTENCY_REVIEW.md`](INCONSISTENCY_REVIEW.md) | **Start here** — flagged issues and decisions needed |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Append-only log and status board |
| [`protocol/MATHEMATICAL_PROTOCOL.md`](protocol/MATHEMATICAL_PROTOCOL.md) | Markdown transcription of the PDF (PDF governs) |
| [`pilot_config.json`](pilot_config.json), [`scene_plan.csv`](scene_plan.csv) | Derived from the protocol (originals not supplied) |
| [`tools/protocol_checks.py`](tools/protocol_checks.py) | Software-only recomputation of every stated number (28/28 pass) |
| [`protocol/DATASET_V2_CONFOUND_REDUCED.md`](protocol/DATASET_V2_CONFOUND_REDUCED.md), [`dataset_v2/`](dataset_v2) | **Pending approval:** 48-clip confound-reduced dataset (no shadow, luminance-matched colours, 12 start positions) |
| [`results/followup_f1_seed_sweep/RESULTS_F1.md`](results/followup_f1_seed_sweep/RESULTS_F1.md) | Follow-up F1: 101 projection seeds (1/101 pass the primary rule) |
| [`results/RESULTS.md`](results/RESULTS.md) | **Results**, tables, caveats; plots, predictions.csv, summary.json, provenance.json beside it |
| [`protocol/DATASET_V1_1_CHANGES.md`](protocol/DATASET_V1_1_CHANGES.md) | Every deviation of the clips from protocol v1.0 (read before approving) |
| [`pilot.py`](pilot.py), [`dataset/`](dataset) | Stage A generator/validator and its 24 clips, manifest, validation, gallery |
| [`sources/`](sources) | Original brief and PDF as received |

## Stages and gate
Stage A (generate + validate 24 FFV1 clips, gallery, contact sheet) → **STOP, user approves clips; approval is recorded against the manifest SHA-256** → Stage B (extract, fit 24 readers, metrics, plots, bundle). Do not run the checkpoint, extract embeddings or report model results before approval. No other seeds, λ, layers, projection sizes, gravities or scenes.

## CLI
Stage B: `python3 stage_b.py selftest|extract|analyze|report` (needs approval receipt; torch CPU + vjepa2 checkout at the pinned revision + checkpoint in `cache/`). Stage A: `pilot.py` —
`generate` · `validate` · `gallery` · `record-approval` — implemented (Stage A). `extract` refuses without a matching approval receipt; `analyze` · `report` — Stage B, not implemented.
Open `dataset/gallery.html` (with `dataset/viewing/` and `dataset/videos/` beside it) or `dataset/contact_sheet.png` to review the clips.

Reproduce the checks: `pip install numpy && python3 tools/protocol_checks.py`

Related earlier work in this repo: [`../docs/jepa-abstraction-experiment/`](../docs/jepa-abstraction-experiment) (separate toy-JEPA plan; untouched).
