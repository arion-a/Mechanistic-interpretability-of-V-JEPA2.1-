# Coding-agent instruction brief

Authoritative companion: MATHEMATICAL_PROTOCOL.md and pilot_config.json. The PDF is the large-type reading copy.

## 10 / Coding-agent brief: staged execution

MISSION: implement this exact 24-video pilot after the user selects the execution environment. Read protocol, pilot_config.json and scene_plan.csv first. Preserve the separate 100-video plan and all synced sources. Do not run the older run.py as if it implements this pilot.

Stage A - implementation and stimulus review

1. Record environment, source revision and dependency versions. Build separate generate, validate, extract, analyze and report commands. Implement the specified formulas with no scientific defaults left implicit.

2. Run software-only checks on hand-computable vectors and analytic trajectories. Generate exactly 24 five-second clips. Validate count, timestamps, visibility, lossless RGB round-trip, identical color-pair trajectories and matching first frames within same-color gravity pairs.

3. Deliver a local gallery listing all 24 clips with scene, gravity and color labels, plus a contact sheet. Ensure every clip can be opened individually. An optional viewing transcode must be labeled derivative and must not enter inference.

4. STOP. Ask the user to inspect and approve the clips. Do not load the checkpoint for inference, extract embeddings, run readers or report empirical model results before this explicit approval. Record the approval with the dataset-manifest hash.

Stage B - only after dataset approval

5. Strictly load the specified encoder and extract 24 early-window embeddings. Masking future frames after full-video encoding is forbidden. The encoder must never receive frames after t=2 seconds.

6. Save the one seeded projection, compute projected embeddings, then fit the 24 readers: 6 folds x 2 representations x 2 tasks. Use training-only scaling, a dual ridge solve and the fixed penalty.

7. Produce every declared metric, table and plot. Report error in metres and storage separately. Do not rerun with a different seed or penalty because the result is poor. A defect correction requires a documented change and rerun of affected outputs.

8. Deliver the reproducible bundle and stop. Do not add gravity values, projection sizes, additional layers, nonlinear readers, significance claims or additional scenes without a new instruction.

## 11 / Required artifacts and acceptance checks

Deliverables

videos/: 24 verified FFV1 clips. gallery.html: all 24 clips with open/download links and contact sheet. manifest.json: IDs, scene parameters, times, colors, codec and SHA-256. dataset_validation.json: explicit checks and failures.

embeddings_full.npy [24,768], embeddings_projected.npy [24,16], projection.npy [16,768], embedding_index.csv, fitted_readers/ with all fold means/scales/weights. provenance.json: exact source, checkpoint hash, dependency versions, device, precision and input frame indices.

predictions.csv: fold, scene, gravity, color, true x/y at 4 s, full/projected/CV/prior predicted x/y, errors, color scores and predicted labels. summary.json: all pooled/group metrics and decision flags; invalid values represented as null with a reason.

Plots: six-group position RMSE comparison; full/projected color correct counts; paired true/predicted future positions; feature payload with overhead clearly separated. Raw plot data must be included. No plot may imply statistical uncertainty that was not estimated.

Software checks before experimental inference

Verify scene counts and exact 20/4 splits; paired physical targets; analytical target positions; constant-velocity error formula; ridge dual/primal agreement on a tiny example; unpenalized intercept; tie-to-orange color rule; aggregate RMSE versus pooled squared errors; zero-denominator handling; reproducible projection hash.

Verify no held-out row contributes to feature means/scales or reader fitting. Ensure target labels and metadata are never embedded into feature arrays. Modifying frames 61-150 must leave the actual tensor passed to the encoder unchanged; test that input-construction property without extra model runs.

Resume only if video hash, input-index list, code hash, config hash, source revision and checkpoint hash match. Do not reuse old embeddings from full-duration videos. Any cached artifact lacking matching provenance must be recomputed after approval.

Acceptance: all 24 clips accounted for; 24 finite full embeddings; 24 finite projected embeddings; exactly one out-of-fold prediction per clip per representation; no omitted groups; every decision reproducible from saved CSV and JSON.

## 12 / Worked checks, references and handoff

Hand-computable scene check (not an inference result)

For scene s=0, x0=-40 m and y0=140 m. At t=4 seconds the true x coordinate is -32 m. The true y coordinate is 61.6 m for g=9.8 and 100.8 m for g=4.9. Both colors have the same target. At t=5 seconds the baseline-gravity y coordinate is 17.5 m, safely above the lower boundary for a radius-five ball.

Interpretation checklist

Distinguish a narrow successful reader from a physics-engine claim. Predicting at two already-seen gravities does not show interpolation to unseen gravity. A linear projection does not identify semantic feature columns. Failure of a linear color reader does not establish irrecoverable color loss. Appearance changes induced by projection are distinct from original JEPA abstraction.

Source anchors

Official architecture and checkpoint source: https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812

Checkpoint: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

Related probing study: Interpreting Physics in Video World Models, arXiv:2602.07050. This pilot is not a reproduction of that paper. Its dataset, ridge settings, projection and decision rules are explicit proposed choices for this small experiment.

Copyable handoff

Implement only the protocol in this package for the selected environment. First generate and validate the 24 colored-ball videos and present every clip for inspection. Stop before inference and wait for my explicit dataset approval. After approval, use the fixed first-two-second input, frozen final-layer JEPA features, seeded 16-D projection, six grouped folds and ridge calculations exactly as specified. Return all metrics, plots, artifacts and provenance, even for a null result. Do not expand the experiment or change its thresholds.

Current status: design documents prepared; no pilot videos generated, no checkpoint inference performed, no readers fitted. The user retains the decision about where to execute.

## Required command interface

Provide commands for generate, validate, extract, analyze and report. Extract must refuse to proceed without a recorded dataset-approval receipt matching manifest SHA-256. No automatic approval flag enabled by default. A dataset edit invalidates the receipt.

## Regression and provenance note

Store source code, environment lock, config and matrix hashes. Reproduce all numerical summaries directly from predictions.csv. Partial failures are reported; no silent filtering of examples. The previous 100-video script is not suitable without explicit reconciliation.
