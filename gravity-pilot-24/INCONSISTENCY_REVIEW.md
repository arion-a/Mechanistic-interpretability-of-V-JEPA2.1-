# Inconsistency & gap review — 24-video gravity pilot (protocol v1.0, 9 Oct 2026)

Reviewed: `sources/MATHEMATICAL_PROTOCOL.pdf` (13 pp) and `sources/CODING_AGENT_BRIEF.md`.
Method: read both in full; recomputed every stated number with `tools/protocol_checks.py` (all 28 checks pass; output in `tools/protocol_checks_result.json`).

**Bottom line:** the mathematics is internally consistent — no numeric errors found. The issues are (a) missing referenced files, (b) one logical redundancy in the baseline rule, and (c) places where the "no scientific default left implicit" requirement is not yet met. **Nothing here has been changed in the protocol; each item needs your decision.**

## A. Needs your decision (could change what is run or concluded)

| ID | Where | Issue | Suggested handling |
|---|---|---|---|
| I-01 | Brief header, §10 | Brief calls `MATHEMATICAL_PROTOCOL.md`, `pilot_config.json` and `scene_plan.csv` "authoritative" / "read first". Only the PDF and brief were supplied. | I reconstructed `pilot_config.json` and `scene_plan.csv` from the PDF (marked DERIVED) and `protocol/MATHEMATICAL_PROTOCOL.md` as a transcription. Send the originals so they can be diffed; until then the PDF is the sole authority. |
| I-02 | §07B, §09 | **"Beat both baselines" is vacuous, and §07B contradicts itself.** E_CV = 16.528 m < E_prior = 19.6 m, and because every scene group contains both gravities the same holds in *each* of the six groups. So `min(E_CV,E_prior)` is always E_CV and the prior baseline can never bind. §07B then says a learned reader "need not outperform constant velocity to be informative" yet "must each beat the smaller of the two baselines" (= CV), and that "only beats constant velocity is insufficient" — impossible, since beating CV already beats the prior. | Decide the intent: (a) accept that the bar is simply E_CV = 16.53 m and reword; or (b) the prior was meant to be a stricter control (e.g. a gravity-aware oracle prior), which would change the design. I will implement the literal rule (`min`) unless told otherwise, and report both baselines separately as required. |
| I-03 | Brief "Required command interface" / "Regression note" | These two sections exist **only in the brief**, not in the PDF, although the brief calls the PDF the "large-type reading copy" of the protocol. The approval-receipt rule therefore has no counterpart in the protocol. | Treat the brief as authoritative for process, PDF for maths. Confirm. |
| I-04 | Brief Stage A.4 | Approval receipt: file name, format and fields are unspecified, and `manifest.json` "times" could change its SHA-256 on regeneration. | Proposed: `approval_receipt.json` with `manifest_sha256`, approver, UTC time, free-text note; manifest must be byte-deterministic (no wall-clock or host fields — put those in `provenance.json`). |

## B. Underspecified (spec requires "no implicit defaults" — I need values or will pin and log them)

| ID | Gap | Why it matters / proposed pin |
|---|---|---|
| I-05 | Source revision `204698b…` could not be verified from this sandbox (GitHub tree URL returned 403 via proxy). The checkpoint URL does resolve (HTTP 200). | Verify revision + record checkpoint SHA-256 at first download; Stage B only. |
| I-06 | "EMA encoder" + "strict load" for a **distilled** ViT-B (`..._dist_vitG_384`): it is not stated which state-dict key holds the weights, nor that an EMA copy exists for a distilled student. Also "hierarchical output disabled" depends on the V-JEPA 2.1 code path. | Inspect checkpoint keys & model code at the pinned revision before extraction; if the premise is wrong that is a blocker to report, not to work around. |
| I-07 | **RESOLVED in Stage A (bgr0).** FFV1 pixel format not specified. Default ffmpeg paths convert RGB→YUV, which breaks the required "decoded RGB equality". | Use an RGB-native format (`bgr0` or `gbrp`) and test round-trip equality on all 151×24 frames. |
| I-08 | **Pinned in Stage A** (Pillow 12.3.0, 1-px grid under ball). Rendering details: grid line width, line draw order vs. ball, Pillow float-bbox rasterisation, Pillow version. | Pin Pillow; 1-px lines, grid under ball; record rendered-frame hashes in manifest. |
| I-09 | `Generator.integers` bit-stream is not guaranteed stable across NumPy versions. | Pin NumPy; reference hash for numpy 2.5.3 is in `pilot_config.json` — compare in the eventual environment; the saved `projection.npy` is canonical. |
| I-10 | Frame source for the encoder: decoded FFV1 vs. in-memory rendered frames. | Decode from the validated FFV1 files (lossless ⇒ identical); log the frame indices. |
| I-11 | Where the 24-clip approval gate lives in `resume` logic ("code hash, config hash …") — no hash algorithm/scope defined. | SHA-256 over git-tracked source + config; logged in provenance. |

## B2. Found during Stage A implementation
* **I-15** Matroska stores timestamps in 1-ms ticks, so "timestamps = k/30 s" can only hold to within 0.33 ms in the file container (frame index k is exact: 151 packets, in order). Validation uses a 0.5 ms tolerance; the source-frame index list, not container time, defines encoder input.
* **I-16** The drawn disc's centroid is quantised by pixel rasterisation: max |error| vs. the analytic centre was 0.566 px (0.295 m), unbiased (mean ≈ 0). My first validation tolerance of 0.5 px was arbitrary and failed; it was replaced by the principled half-pixel-per-axis bound (√2/2 px) plus a separate bias check — not tuned to pass. This pixel-level noise (~0.3 m) is far below the 16.5 m CV baseline but should be remembered if centres are ever read back from pixels.

## C. Minor / cosmetic

* **I-12** "Five-second clips" vs 151 frames = 5.0333 s playback (5.0 s simulated). Label clips "5 s simulated, 151 frames" everywhere.
* **I-13** Section numbering "07B"; PDF footer reads "Design only" and §1 says only document preparation is authorised — fine for the original task, but ensure the Stage A authorisation is stated in the log (done: `EXPERIMENT_LOG.md`).
* **I-14** The brief and the repo's earlier work refer to an "older `run.py`" and a "separate 100-video plan". **Neither exists in this repository.** The repo holds (i) `docs/jepa-abstraction-experiment/` (a from-scratch toy-JEPA plan) and (ii) two 3D-V-JEPA `.docx` files (PyBullet, 6 s, 180 frames at 1/240 s). Nothing conflicts with this pilot, but there is nothing named "100-video" to preserve; I left all existing files untouched.

## D. Observations (not errors — read before interpreting results)

* **O-1** The 1/√16 factor in R is cancelled by the scalar-RMS scaling (§05), so it affects storage/hash only, never the readers (check in `protocol_checks.py`).
* **O-2** With n = 20 training clips and 768 features, λ = 0.01 gives a near-interpolating (minimum-norm) full-embedding reader; the 16-D reader is rank ≤ 16 < 20. The two representations therefore differ in regime, not just dimension — consistent with the protocol's "descriptive" framing, but worth stating in the report.
* **O-3** The CV baseline is deliberately an oracle on exact simulator centres (labelled so in the protocol). A "G_r > 0" win is over an oracle, so it is a *strong* bar; a miss does not mean the encoder lacks information about motion.
* **O-4** Folds hold out a grid cell, but every x0 and y0 value still appears in training — this is interpolation within the grid, not extrapolation to new coordinates.
* **O-5** Colour is not luminance-matched (stated); any colour effect is confounded with brightness.

## Verified-correct items (no action)

Scene layout and splits (6×2×2, 20/4, 10/10 colours, 10/10 gravities per fold); frame indices/timestamps; 4608 = 8×24×24; pixel mapping and ball visibility (u∈[115.2,288], v∈[76.8,350.4]); target formula and worked example (−32, 61.6 / 100.8); CV error 32g/15 = 20.9066667 / 10.4533333; E_CV = 16.5282 m; prior (8, −58.8), error 19.6 m; storage 3072/64 B, 48×, 97.9167 %, 73,728 / 1,536 / 49,152 / 50,688 B; dual = primal ridge and unpenalised intercept.
