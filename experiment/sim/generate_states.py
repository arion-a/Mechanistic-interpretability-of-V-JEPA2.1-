"""Phase 1 of the parallelized Blender pipeline: physics-only state generation.

Runs PyBullet simulation for N base worlds x 9 clips (baseline + 8
interventions), WITHOUT rendering, and applies the observability (in-frame)
gate. This runs in regular system Python because pybullet is not available
inside Blender's bundled interpreter - Blender only handles phase 2
(rendering from these saved state arrays, see remote/blender_render_worker.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pybullet as p

from world import baseline_and_interventions, run_episode, sample_anchor, BASELINE_G, BASELINE_E, BASELINE_MU
import render as render_mod

ROOT_SEED = 42
SPHERE_RADIUS_TOL = 0.16


def in_frame_mask(position: np.ndarray, image_size: int = 256) -> np.ndarray:
    """NDC-space visibility check (resolution-independent - only depends on FOV/aspect)."""
    V = np.array(render_mod._VIEW_MATRIX).reshape(4, 4, order="F")
    P = np.array(render_mod._PROJ_MATRIX).reshape(4, 4, order="F")
    ones = np.ones((position.shape[0], 1))
    homog = np.concatenate([position, ones], axis=1)
    clip = homog @ V.T @ P.T
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None]
    return np.all(np.abs(ndc[:, :2]) < 1.0, axis=1) & (ndc[:, 2] > -1) & (ndc[:, 2] < 1) & (w > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=50)
    ap.add_argument("--out", type=str, default="../data/blender_pilot")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    state_dir = os.path.join(out_dir, "state")
    os.makedirs(state_dir, exist_ok=True)

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(__import__("pybullet_data").getDataPath())

    specs = baseline_and_interventions()
    manifest_rows = []
    t_start = time.time()
    rejected = []

    for world_idx in range(args.n_worlds):
        anchor = sample_anchor(ROOT_SEED, world_idx)
        for spec in specs:
            episode = run_episode(anchor, spec, client, render=False)

            min_z = float(episode.position[:, 2].min())
            visible = bool(in_frame_mask(episode.position[episode.frame_steps]).all())
            accepted = (min_z > -SPHERE_RADIUS_TOL) and visible

            clip_id = f"world{world_idx:03d}_{spec.name}"
            state_path = os.path.join(state_dir, f"{clip_id}.npz")
            np.savez_compressed(
                state_path,
                position=episode.position,
                frame_steps=episode.frame_steps,
                contact_flags=episode.contact_flags,
            )
            manifest_rows.append({
                "clip_id": clip_id, "world_index": world_idx,
                "intervention": spec.name, "factor": spec.factor,
                "g": spec.g, "e": spec.e, "mu": spec.mu, "velocity_delta": spec.velocity_delta,
                "accepted": accepted, "min_z_m": min_z,
                "n_contact_steps": int(episode.contact_flags.sum()),
                "state_path": os.path.relpath(state_path, out_dir),
            })
            if not accepted:
                rejected.append(clip_id)

        if world_idx % 10 == 0:
            print(f"world {world_idx}/{args.n_worlds} done ({time.time()-t_start:.1f}s)", flush=True)

    p.disconnect(client)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({
            "root_seed": ROOT_SEED, "n_worlds": args.n_worlds,
            "n_clips": len(manifest_rows), "n_rejected": len(rejected),
            "rejected_clip_ids": rejected,
            "baseline": {"g": BASELINE_G, "e": BASELINE_E, "mu": BASELINE_MU},
            "wall_time_s": time.time() - t_start,
            "rows": manifest_rows,
        }, f, indent=2)

    print(f"Phase 1 done: {len(manifest_rows)} clips ({len(rejected)} rejected) "
          f"in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
