"""Local, GPU-free diagnostic: how much does each factor actually move the
*physical trajectory* (position/velocity over time), independent of V-JEPA?

If a factor's trajectory divergence from baseline is tiny compared to the
others', that's evidence any weak embedding-level result for it reflects a
real physical-effect-size problem in the scene design, not a representational
failure of the encoder. Run before investing in any representational
follow-up (alternative bases, sparse autoencoders) for a factor that looks
weak at the embedding level.

Usage: python3 diagnose_trajectory_divergence.py [--n-worlds 30] [--root-seed 42]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from world import sample_anchor, baseline_and_interventions, run_episode  # noqa: E402


def trajectory_metrics(ep_baseline, ep_intervention) -> dict:
    """Summarize how far an intervention's trajectory diverges from baseline."""
    pos_b = ep_baseline.position
    pos_i = ep_intervention.position
    disp = np.linalg.norm(pos_i - pos_b, axis=1)  # per-step Euclidean displacement, meters
    vel_disp = np.linalg.norm(ep_intervention.linear_velocity - ep_baseline.linear_velocity, axis=1)
    return {
        "mean_pos_disp_m": float(disp.mean()),
        "max_pos_disp_m": float(disp.max()),
        "final_pos_disp_m": float(disp[-1]),
        "mean_vel_disp_mps": float(vel_disp.mean()),
        "n_contacts_baseline": int(ep_baseline.contact_flags.sum()),
        "n_contacts_intervention": int(ep_intervention.contact_flags.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-worlds", type=int, default=30)
    parser.add_argument("--root-seed", type=int, default=42)
    args = parser.parse_args()

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    specs = baseline_and_interventions()
    baseline_spec = next(s for s in specs if s.factor == "baseline")
    factor_specs = [s for s in specs if s.factor != "baseline"]

    results = {s.name: [] for s in factor_specs}
    for world_idx in range(args.n_worlds):
        anchor = sample_anchor(args.root_seed, world_idx)
        ep_baseline = run_episode(anchor, baseline_spec, client, render=False)
        for spec in factor_specs:
            ep = run_episode(anchor, spec, client, render=False)
            results[spec.name].append(trajectory_metrics(ep_baseline, ep))

    print(f"{'intervention':<20} {'factor':<12} {'mean_pos_disp(m)':>18} {'max_pos_disp(m)':>18} "
          f"{'final_pos_disp(m)':>18} {'mean_vel_disp(m/s)':>20} {'contacts(base->interv)':>24}")
    factor_summary: dict[str, list[float]] = {}
    for spec in factor_specs:
        rows = results[spec.name]
        mean_pos = np.mean([r["mean_pos_disp_m"] for r in rows])
        max_pos = np.mean([r["max_pos_disp_m"] for r in rows])
        final_pos = np.mean([r["final_pos_disp_m"] for r in rows])
        mean_vel = np.mean([r["mean_vel_disp_mps"] for r in rows])
        contacts_b = np.mean([r["n_contacts_baseline"] for r in rows])
        contacts_i = np.mean([r["n_contacts_intervention"] for r in rows])
        print(f"{spec.name:<20} {spec.factor:<12} {mean_pos:>18.5f} {max_pos:>18.5f} "
              f"{final_pos:>18.5f} {mean_vel:>20.5f} {contacts_b:>10.1f} -> {contacts_i:<10.1f}")
        factor_summary.setdefault(spec.factor, []).append(mean_pos)

    print("\nPer-factor average mean_pos_disp across both doses (bigger = physics itself diverges more):")
    for factor, vals in factor_summary.items():
        print(f"  {factor:<12} {np.mean(vals):.5f} m")


if __name__ == "__main__":
    main()
