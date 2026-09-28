# Results & Data Reporting Specification

Authoritative metrics, uncertainty, and an honest result narrative

# 1. Reporting principles
Every reported number is traceable to a protocol version, dataset hash, checkpoint, evaluation suite, and aggregation version. CSV/JSON aggregates are authoritative; Excel is a review and visualization layer. No experimental data are included in this package. Blank template cells mean no result has been imported.
The primary report begins with what was tested and what the data support. Separate computational completion, metric eligibility, statistical evidence, and interpretation. A healthy trained model can yield an invalid ARS if the state–oracle gap is too small. An invalid task cannot establish absence of abstraction.
# 2. Required metric dictionary
| Metric | Definition / unit | Required breakdown |
| L_S, L_J, L_O | Mean squared 3D displacement divided by 1 m²; time/law/group weights fixed | condition, suite, corpus, training seed; raw group losses retained. |
| G / usefulness | L_S − L_O, normalized MSE | point estimate and paired bootstrap CI; denominator eligibility. |
| B / history benefit | L_S − L_J, normalized MSE | point estimate, CI, contrast and multiplicity family. |
| ARS | (L_S−L_J)/(L_S−L_O), unitless, unclipped | ratio-of-means estimate, hierarchical CI, validity status. |
| Endpoint RMSE | sqrt(mean squared Euclidean endpoint error), meters | same group weighting, bootstrap interval. |
| Velocity RMSE | sqrt(mean squared Euclidean velocity error), m/s | optional diagnostic if velocity readout is implemented; not substitute for primary target. |
| Paired divergence error | MSE of predicted minus true between-law trajectory differences | matched anchor/law pair; group-clustered CI. |
| Inferability | gravity MAE m/s²; R²; balanced accuracy for discrete laws | analytic tracked-state and supervised visual estimators separately. |
| Transfer gain | target-task state loss minus frozen-history loss; target-task ARS if eligible | phenomenon, source-target mode, readout-label budget. |
| Layer representation | gravity probe R²/MAE; nuisance probe; rank fraction; swap benefit | layer, probe type, split, multiplicity treatment. |
| Frontier | ARS interval and RMSE interval versus stress; bracketed crossing | model, suite, stress axis/side/level; success/uncertain/fail/ineligible. |
| Cost | parameters, clips/tokens seen, optimizer updates, FLOP estimate, GPU-hours, peak memory | hardware, precision, batch/accumulation; pretrain/readout/eval separately. |
| Reliability | planned/completed/accepted/failed counts; retry and rejection rates | condition, failure cause, data and code version. |
Metric IDs are stable, e.g. position_nmse_v1, endpoint_rmse_m_v1, ars_v1. Changing weighting, units, baseline, oracle information, or target window requires a new metric version. Do not pool metric versions.
# 3. Metadata contracts
Experiment registry: experiment_id, protocol_version, stage, hypothesis_id, analysis_role, condition_id, comparison_family, config_id, dataset_plan_id, planned_corpora, seeds_per_corpus, target_suite, status, preregistered_utc, amendment_id, owner, notes. Allowed analysis_role values: pilot, screening, confirmatory, exploratory.
Run metadata: run_id, experiment_id, config_id, dataset_id, corpus_id, train_seed, readout_seed, status, attempt, parent_run_id, git_commit, environment_hash, start_utc, end_utc, gpu_model, gpu_count, precision, optimizer_steps, clips_seen, tokens_seen, estimated_flops, gpu_hours, peak_memory_gb, checkpoint_uri, failure_code. Record actual values, not just requested settings.
Dataset metadata: dataset_id, generator_version, manifest_hash, split, phenomenon, corpus_id, n_clips, n_groups, law_regime, gravity_values_or_distribution, law_variance, persistence_frames, dt_s, fps, frames_per_clip, resolution, state_units, camera_spec_hash, nuisance_policy, rejection_fraction, checksum_uri, created_utc. For variable interval data record time stamps, not just fps.
Model metadata: config_id, architecture_version, depth, width, heads, mlp_ratio, patch_size, temporal_patch, context_frames, horizon_frames, target_window_definition, predictor_depth/width, parameters_encoder/total, EMA schedule, objective_version, optimizer config hash, initialization, regularization, batch_effective, learning-rate schedule, normalization, pooling, pretrained_data_id (null from scratch).
Aggregate provenance additionally records analysis_id, included_run_ids_uri, exclusion_log_uri, bootstrap_seed, bootstrap_replicates, corpus_count, training_seed_count, eval_corpus_count, matched_group_count, estimand, confidence_method, multiplicity_family, adjusted_p_value, practical_threshold, status, and pipeline_git_commit.
# 4. Required tables and plots
| Output | Contents | Guardrail |
| Table 1: design/completion | Planned and accepted runs per cell; failed/invalid counts; resources | All registered cells remain visible. |
| Table 2: primary endpoint | L_S/L_J/L_O, G, B, ARS, CIs, corpus/seed counts, status | ARS omitted when gap ineligible; raw losses remain. |
| Table 3: confirmation contrasts | Effect, CI, raw and Holm-adjusted p, preregistered threshold | No stars without numeric effects and analysis family. |
| Table 4: transfer | Phenomenon, adaptation labels, controls, transfer gain, uncertainty | Frozen versus fine-tuned and zero-shot versus adapted explicit. |
| Table 5: costs/frontiers | Compute, data, parameters, boundary brackets, failure cause | Separate task-ineligible from model failure. |
| Figure 1 | Matched-state histories, identical anchor, divergent futures | Same visual scale; law revealed only in evaluator annotations. |
| Figure 2 | Raw-loss and ARS versus horizon/context with CIs | Gap/eligibility panel accompanies ratio. |
| Figure 3 | ARS versus diversity and persistence; usefulness/inferability panels | Show law variance and realized dwell. |
| Figure 4 | ARS versus depth, width, parameters, data, compute | Matched controls identified; no unsupported power-law claim. |
| Figure 5 | Frozen transfer learning curves and source-history shuffle control | Same label and optimization budget. |
| Figure 6 | Layerwise probes and intervention effects | Label exploratory or multiplicity-corrected. |
| Figure 7 | Context×horizon and noise×dwell phase-style grids | Distinguish valid values, ineligible, uncertain, and missing. |
| Figure 8 | Frontier stress curves with interval bands and oracle usefulness | Boundaries shown as brackets, not exact constants. |
Use accessible palettes, named axes with units, error-bar definitions, number of independent corpora/seeds, and suite/version in captions. Show seed points where readable. Never smooth away negative ARS or clip values above one. Logarithmic axes must identify zero/unsupported levels explicitly.
# 5. Uncertainty and aggregation procedure
The statistics pipeline computes 10,000 paired hierarchical bootstrap draws from group-level losses using the hierarchy in the research plan. Every draw computes the ratio after averaging losses. Export gap_ci_low/high, benefit_ci_low/high, ars_ci_low/high, endpoint_rmse_ci_low/high, invalid_ratio_fraction, and confidence_method. Excel does not derive these intervals from run SD.
If fewer than the preregistered corpus or seed counts complete, mark the confirmatory analysis incomplete and show its actual sample size. Do not silently replace failures with successful seeds. A sensitivity analysis may include documented replacements under the same failure policy, with original failures still reported.
Report within-corpus training-seed SD and between-corpus SD separately; pooled run SD is descriptive. For contrast intervals pair the same corpus IDs, seed IDs and evaluation groups wherever the design permits. Unequal/missing pairs require a documented estimator, not row-order subtraction.
No analytic normal CI is supplied for ARS because it is a ratio with a potentially weak denominator. A bootstrap interval with a nonpositive denominator draw is flagged UNSTABLE_RATIO and suppressed as a clean finite inferential result. Raw loss and benefit intervals remain useful. CI endpoints imported into Excel must satisfy low≤high and include the stated method, though an estimate need not always lie within a percentile interval.
# 6. XLSX workbook guide
The ten requested sheets are included. Input cells use blue text; formula cells have a pale teal fill; headers are navy; blank rows are intentional. Each input sheet has an Excel table, filters, frozen headers, a units/usage note, and 300 preallocated record rows. No sheet contains fabricated research outcomes.
| Sheet | Row grain | Main use |
| Experiment Registry | one registered cell | Scope, hypotheses, planned replication, status. |
| Runs | one execution attempt | Actual runtime, seed, artifact and failure tracking. |
| Dataset Metadata | one dataset split/replica | Simulation, laws, splits, hashes. |
| Model Configs | one resolved model/training configuration | Architecture and controlled resource settings. |
| Primary Metrics | one run evaluation OR one explicit condition aggregate | Physical losses, ARS formulas, imported bootstrap CIs. |
| Layerwise Metrics | one layer/probe/suite/analysis record | Conditional/nuisance probes and swap tests. |
| Scaling Sweeps | one condition aggregate per scaling comparison | Factor values, compute, ARS interval. |
| Breaking Frontier | one model/axis/level aggregate | Interval-based operational classification. |
| Checkpoints | one phase gate | Owner, evidence, state, decision and next action. |
| Summary Dashboard | fixed filtered views | Completion, four sweep charts, context×horizon heat map. |
Dashboard selectors: analysis_id, suite_id, metric_id, target reference context/horizon, K, persistence, depth, width, training clips, compute_multiplier, and epsilon. Four curves vary one factor (horizon, K, depth, N) while holding all others to selectors. The heat map varies c/h with remaining factors fixed. Only VALID confirmatory aggregates qualify. If two rows match one plotted cell, display a duplicate marker/chart gap rather than average them.
Point charts are navigation aids; authoritative imported CIs are visible alongside summary values and in Primary Metrics. Charts do not replace final publication figures with bootstrap error bars. A phase-style heat map is descriptive; a colored cell is not proof of a phase transition.
Excel computes G, B, ARS, CI width, and frontier status. CI fields are populated by the external statistics pipeline. Missing or invalid rows remain blank; charts use NA() to create gaps. On opening, Excel is asked to recalculate. Rendering and cached formula values may vary in other spreadsheet applications.
# 7. Import and validation workflow
1. Validate exported CSV/JSON against contracts, unique keys, units, accepted run IDs and aggregate counts.
2. Compute uncertainty externally from per-group data, keeping group/corpus pairing. Export only approved aggregate rows to confirmatory summaries.
3. Fill input columns by exact snake_case header, preserving formula columns, tables and chart ranges. The supplied importer implements this mapping; it accepts a directory of sheet-named CSV files.
4. Reopen/recalculate in Excel or another compatible spreadsheet engine, then compare point estimates and CI fields against authoritative exports. Formula presence is not equivalent to evaluated numerical correctness.
5. Save a release copy with analysis_id and protocol version. Include the CSV/JSON, workbook, figures and a hash manifest together. Never use the dashboard as the sole scientific record.
# 8. Reusable results memo template
Title: JEPA abstraction-emergence experiment — [analysis_id / protocol version]
Question and scope: We tested whether [training condition] changes reusable law information in [synthetic phenomena], using [context/horizon/law regimes]. The primary endpoint was [metric/version] on [locked suite].
Design and completion: [planned] runs across [corpora] independent corpora and [seeds] initialization seeds per corpus; [accepted] accepted, [failed] failed, [invalid] ineligible. Exclusions and amendments: [links/reasons].
Primary evidence: L_S=[ ], L_J=[ ], L_O=[ ]; gap=[ ] [95% CI]; history benefit=[ ] [95% CI]; ARS=[ ] [95% CI/status]. Prespecified contrast=[ ] [CI], adjusted p=[ ], practical threshold=[ ].
Positive result wording: “Under the tested persistent-law conditions, frozen JEPA history features recovered [fraction] of the state-to-oracle prediction gap. The prespecified [transfer] control also improved by [effect/CI]. This supports reusable predictive information about gravity within these tasks; it does not establish a general causal concept.”
Negative result wording: “With an eligible oracle gap and validated optimization, JEPA history features performed worse than [comparison] by [effect/CI]. The failure occurred under [scope]. We cannot distinguish [remaining alternatives] without [next test].”
Practically null wording: “The interval for the prespecified benefit lies within the preregistered equivalence band [−.10,.10] in ARS units. This supports a practically small effect in this regime, subject to [finite-cluster/readout limitations].”
Inconclusive wording: “The interval includes both no benefit and meaningful benefit / the oracle gap was ineligible / the run set was incomplete. The experiment does not currently distinguish the competing explanations.” Select the actual reason; do not call a wide nonsignificant result proof of no abstraction.
Mechanism checks: inferability=[ ]; transfer=[ ]; cause-disambiguation=[ ]; shuffled/random controls=[ ]; leakage/collapse checks=[ ]. State which evidence-ladder level was reached.
Limitations and next decision: [supported domain], [failed frontier], [compute/replication limits], [one discriminating follow-up]. Include links to aggregate files, run IDs, dataset/model hashes, and amendments.
# 9. Release checklist
All reported results are real imported measurements; fixture values are isolated. Counts match the registry. Metric units and weights match protocol. Gap eligibility precedes ARS interpretation. CI hierarchy and multiplicity are stated. Transfer supervision is disclosed. Failures, nulls and negative values are retained. Figures and workbook resolve to immutable data. No conclusions are implied by the blank template.
