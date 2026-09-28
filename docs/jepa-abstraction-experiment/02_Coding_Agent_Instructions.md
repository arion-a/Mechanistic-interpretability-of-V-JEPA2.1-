# Coding Agent Instructions

Interfaces, tests, ownership, and release criteria

# 1. Operating contract
Build the protocol in 01_Research_Experiment_Plan. Do not change the scientific endpoint, split policy, baseline information, or seed hierarchy to make a run succeed. Submit a protocol amendment with reason, affected runs, and a new protocol version. Retain all earlier outputs and failure records.
Roles below are implementation responsibilities, not a request to launch agents automatically. One person or agent may own several roles, but interfaces and independent checks remain explicit. The scientific lead owns protocol freeze; statistics owns estimands and uncertainty; evaluation owns access to locked test labels.
# 2. Roles and exact handoffs
| Owner | Inputs | Outputs | Acceptance before handoff |
| Simulator | world.yaml; scene_seed; nuisance_seed; anchor state; law schedule | episode.npz: time_s[T], state[T,6], gravity[T], RGB uint8[T,96,96,3]; episode.json with units/support/version | Analytic free-flight error <1e−6 m; anchor equality <1e−6; common support; contact-specific validation. |
| Dataset | simulator release; split manifest; dataset.yaml; corpus seeds | immutable shard files; manifest.parquet; dataset.json; SHA256SUMS; audit.json | No group/asset leakage; checksums valid; loader excludes hidden labels; full split counts. |
| JEPA | RGB contexts/targets and train config | encoder/predictor/EMA checkpoints; optimizer and RNG state; training.jsonl; model.json | Future isolation test; EMA/gradient tests; no collapse; exact resume in supported deterministic environment. |
| Experiment runner | expanded registry; config hashes; resource budget | immutable run directory; events.jsonl; run.json; exit status; runtime profile | Unique IDs, atomic completion, budget stop, no overwritten runs; all failures visible. |
| Evaluation | frozen checkpoint; frozen readout; locked suite manifest; baseline/oracle code | per_group.parquet; primary_metrics.csv; predictions.npz; evaluation.json | Common target/weights; paired group coverage; oracle residual verified; eligibility flags applied. |
| Representation analysis | frozen layer features; analysis split manifests | layerwise_metrics.csv; probe checkpoints; swap outputs; analysis.json | No probe split leakage; state/nuisance controls; layer selection only on validation. |
| Scaling sweeps | registered Stage C cells; pilot profile | scaling_sweeps.csv; cost fit; Pareto plots; fit residuals | Tokens/params/FLOPs accounted; restricted fits; uncertainty and extrapolation labeled. |
| Breaking frontier | fixed models; stress configs; suite version | breaking_frontier.csv; per-level metrics; brackets; stress audit | Oracle recomputed; missing/uncertain separated from failure; repeated boundary levels. |
| Statistics | per-group losses; nested seed IDs; locked contrast list | bootstrap_draws.parquet; summary_metrics.csv; contrast_tests.csv; stats.json | Ratio-of-means, paired resampling, denominator gate, multiplicity, synthetic-fixture recovery. |
| Reporting/dashboard | validated CSV/JSON; summary_metrics; checkpoint records | filled XLSX; plots; results memo; export manifest | No mock values; workbook/CSV agreement; unresolved failures prominent; provenance linked. |
Every handoff includes schema_version, protocol_version, git_commit, config_sha256, input artifact hashes, producer, created_utc, acceptance report, and limitations. Consumer rejects incompatible schema versions and hash mismatches. A successful process exit is not sufficient acceptance.
# 3. Repository and artifact layout
```text
jepa-abstraction/
  README.md
  configs/{world,dataset,model,train,eval,stages}/
  configs/protocol_v1.json
  src/{sim,data,models,runner,eval,analysis,stats,reporting}/
  tests/{unit,integration,scientific,fixtures}/
  contracts/{schemas,metric_dictionary,seed_policy}/
  registry/experiments.jsonl
  data/<dataset_id>/{dataset.json,manifest.parquet,shards/,audit.json}
  runs/<run_id>/{config.json,run.json,events.jsonl,checkpoints/,metrics/}
  evaluations/<evaluation_id>/{per_group.parquet,predictions.npz,evaluation.json}
  aggregates/<analysis_id>/{summary_metrics.csv,bootstrap_draws.parquet,stats.json}
  reports/<release_id>/{dashboard.xlsx,results.docx,figures/,manifest.json}
```
Store large data/checkpoints in an artifact store with content hashes; repository contains code, configs, manifests, and small fixtures. No absolute user-specific paths in portable configs. Paths in logs may be relative artifact URIs. Never commit credentials or hidden test labels in training fixtures.
# 4. Configuration and seed conventions
Use schema-validated JSON as canonical config; YAML may be authored but is resolved to JSON before execution. Reject unknown keys. Expand every default and record resolved values. Canonical hash is SHA-256 of UTF-8 JSON with sorted keys, no whitespace, allow_nan=false; exclude only runtime fields explicitly listed in schema. Units must be part of field names or schema definitions.
IDs: experiment_id=human-readable registered condition; config_id=first 16 hex characters of canonical config hash; dataset_id=generator/version/split-manifest hash; run_id=config_id-corpusNN-seedNN-attemptNN; evaluation_id=run_id-suite_id-readout_seed. A retry is a new attempt linked by parent_run_id; aggregators use one accepted attempt per logical run.
Derive each 32-bit seed from SHA-256(root_seed|protocol_version|role|replica|index), first four digest bytes, big endian. Roles include simulator, renderer, split, corpus, initialization, sampler, augmentation, readout, bootstrap. Never use Python hash(), process ID, wall clock, or unordered filesystem iteration. Paired conditions reuse explicitly recorded seed IDs, not hidden global RNG state.
Seed Python, NumPy, framework CPU/CUDA and worker generators. Record GPU, driver, libraries, precision and deterministic flags; use deterministic kernels where available. Bitwise reproducibility is guaranteed only in a pinned supported environment. Else declare numerical tolerance and verify scientifically equivalent reruns. Store RNG and sampler states in resumable checkpoints.
# 5. Dataset and prediction interfaces
Episode manifest required fields: episode_id, group_id, anchor_family_id, corpus_id, split, phenomenon, asset_family_id, law_regime, law_schedule_uri, nuisance_seed, sim_seed, render_seed, rgb_uri, state_uri, checksum, anchor_index, fps, dt_s, n_frames, accepted, rejection_reason. The training loader receives a filtered view with only RGB URI, indexing, group/split and sampling fields; the raw manifest is evaluator-only.
Prediction arrays: prediction[evaluation_id,episode_id,offset,xyz], target with same axes, validity mask and metric weights. Store float32 predictions; compute physical losses in float64. Do not drop difficult examples silently; an invalid prediction flags the evaluation and records count and cause. All baselines must cover identical valid group sets.
Per-group loss table fields: evaluation_id, condition_id, train_corpus_id, train_seed, eval_corpus_id, group_id, suite_id, split, law, horizon_frames, n_targets, loss_state, loss_jepa, loss_oracle, endpoint_sqerr_jepa_m2, gravity_abs_error, weight. Losses are first averaged within law/anchor as specified; export the law rows or separate episode table to preserve audits. A group aggregate weights laws equally, independent of video length.
# 6. Logging and failure handling
events.jsonl fields: schema_version, event_id, utc, run_id, stage, event_type, step, epoch, train_loss, val_loss, lr, ema_decay, grad_norm, feature_std_median, effective_rank_fraction, clips_seen, tokens_seen, estimated_flops, gpu_seconds, peak_memory_bytes, checkpoint_uri, status, error_code, message. Missing measurements are null, not zero or NaN. Record observation units in the metric dictionary.
Status transitions: PLANNED → RUNNING → COMPLETED or FAILED/CANCELLED. Evaluation separately has VALID, INVALID_GAP, UNSTABLE_RATIO, INVALID_DATA, INVALID_MODEL, or PENDING. A scientifically invalid evaluation can belong to a computationally completed run. Checkpoints have PENDING, PASS, HOLD, FAIL; only the scientific owner records PASS.
On NaN/Inf, nonfinite gradients, corruption, or leakage: stop, save diagnostic bundle, record last good checkpoint, quarantine results. Infrastructure retries: at most two, same logical seed and config, new attempt ID. OOM may use gradient accumulation only if effective batch, schedule and numerics contract is preserved; otherwise new config. Never reduce resolution/context silently. Resume only after checksums and complete optimizer/RNG state verify.
Write to temporary paths, fsync where supported, then atomically rename and create a completion marker. An output lacking a marker is incomplete. Before aggregating, require exactly one accepted attempt, expected groups, and registered provenance. Missing runs remain in the denominator of completion reports but not numerical estimates; report counts and reasons by cell.
# 7. Mandatory tests
| Test family | Required checks | Release gate |
| Simulator unit | Gravity sign/units; analytic position/velocity; time indexing; anchor equality; integration convergence | All tolerances met across extreme supported parameters. |
| Dataset unit | Deterministic generation; checksum stability; split disjointness; no label/nuisance correlations; group-level rejection | Zero group overlap; correlation audit reviewed, not inferred from a single p-value. |
| Model unit | Shapes across c/h/depth/width; target stop-gradient; correct EMA; future-perturbation leaves context z unchanged | No target gradients or future influence. |
| Runner integration | 10-step train/save/resume; interrupted shard write; OOM/failure injection; duplicate IDs | Correct state recovery and no silent duplicate acceptance. |
| Scientific fixtures | Constant-gravity analytic oracle; law-ambiguous history; exact matched-state different futures; switched-law oracle | Expected identifiability and Bayes-loss behavior. |
| Metric unit | Perfect oracle gives ARS=1; state copy gives 0; worse JEPA negative; small gap invalid; mean ratios differs from ratio means | Reference values reproduced; no clamping or invalid finite CIs. |
| Statistics | Nested clusters and paired contrasts; known-effect simulation; null calibration; missingness handling | Coverage checked by simulation with finite-sample tolerance documented. |
| Reporting integration | CSV→XLSX→readback; formula presence; missing remains blank; no DIV/0; charts point to summaries | Dashboard agrees with authoritative aggregate JSON after recalculation. |
| End-to-end pilot | Generate tiny data, train all three readouts, evaluate, bootstrap, publish dry-run report | Complete provenance, no leakage, no simulated values mixed with results. |
Collapse pilot trigger: median latent dimension SD <1e−3 on fixed validation examples, or effective-rank fraction <.05 for three evaluations. These are diagnostic defaults to calibrate once against random and healthy encoder references; passing them alone does not establish healthy learning.
# 8. Command interface and completion definition
Required CLI interface (to be implemented): sim validate --config; data build --config; train run --config; eval run --run-id --suite; stats aggregate --registry --analysis-config; report build --analysis-id. All commands accept --dry-run and --output-dir, return nonzero on failure, and write machine-readable acceptance reports. This package supplies contracts, not these training commands.
A role is done when outputs pass schema checks, scientific tests and consumer acceptance, a reproducible command is recorded, resource costs are logged, and known limitations are explicit. No role may declare the central hypothesis supported based solely on its local metric.
# 9. Dashboard integration contract
CSV templates mirror XLSX input columns. Use stable IDs and snake_case names; units are explicit. Formula-derived columns are excluded from CSV templates. Use UTF-8 RFC 4180 CSV, decimal points, ISO-8601 UTC timestamps, empty fields for missing; booleans true/false; no locale thousands separators. Treat imported strings beginning with =, +, −, or @ as text to prevent spreadsheet formula injection.
Primary Metrics mixes run and aggregate rows only through explicit record_type. Dashboard uses record_type=aggregate with analysis_role=confirmatory and metric_status=VALID, plus exact analysis_id/suite/metric filters. It never averages per-seed ARS. One aggregate row per condition per analysis; duplicate keys are a hard error. Hierarchical bootstrap runs outside Excel; export its confidence limits and status to the workbook.
Workbook row capacity is 300 data records per input sheet. If expanding, extend tables, formulas, validation and summary references together. Charts are tied to summary blocks; update these blocks for new factor levels. Blank template formulas intentionally return empty strings or chart gaps. Excel/compatible recalculation is required; stored authoritative numerical outputs live in CSV/JSON.
