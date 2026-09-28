# final_pilot data archive (partial)

This is what was downloaded from the GPU pod before it was shut down, saved
here because `experiment/.gitignore` excludes `data/` (regenerable
simulation/render/encoding output isn't normally committed).

- `manifest.json` — the full phase-1 physics manifest: all 25 worlds x 9
  clips = 225 rows (0 rejected by the visibility gate), with per-clip
  physics settings (g/e/mu/velocity_delta) and `state_path` pointers.
- `features/world000_baseline.npy`, `features/world000_gravity_high.npy` —
  2 of the 225 V-JEPA 2 feature vectors (32,768 floats each), kept as the
  worked example the methods report (`experiment/dashboard/methods_report.html`)
  computes its numbers from.

**Not preserved**: the other 223 feature vectors, all 225 per-clip state
`.npz` files (position/frame_steps/contact_flags), and `features_manifest.json`
(encode-run metadata) — these were never bulk-downloaded off the pod and no
longer exist now that it's off. The full aggregate analysis derived from all
225 clips *is* preserved, in `experiment/results/final_pilot/pilot_analysis.json`
and `specificity_matrix.csv` (git-tracked, not gitignored, computed by
`experiment/analysis/svd_analysis.py` while the pod was still up).

To regenerate the full dataset from scratch: rerun the pipeline described in
`experiment/remote/` (phase 1: `sim/generate_states.py`, phase 2:
`remote/blender_render_worker.py` + `remote/encode_and_sweep.py`) with
`root_seed=42`, `n_worlds=25` — the simulation is fully deterministic, so it
reproduces the same 225 clips exactly.
