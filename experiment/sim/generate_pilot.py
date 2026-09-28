"""Generate a small engineering-pilot dataset: N base worlds x (baseline + 8 interventions).

This is a smoke-test scale run (PyBullet TinyRenderer, handful of worlds), not
the confirmatory 1,200-world / 16,560-clip study specified in the agenda doc,
which requires Blender EEVEE rendering and a much larger compute budget.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pybullet as p

from world import (
    baseline_and_interventions,
    run_episode,
    sample_anchor,
    BASELINE_G, BASELINE_E, BASELINE_MU,
)
import render as render_mod


def in_frame_mask(position: np.ndarray) -> np.ndarray:
    """NDC-space visibility check against the fixed camera in sim/render.py."""
    ones = np.ones((position.shape[0], 1))
    homog = np.concatenate([position, ones], axis=1)
    V = np.array(render_mod._VIEW_MATRIX).reshape(4, 4, order="F")
    P = np.array(render_mod._PROJ_MATRIX).reshape(4, 4, order="F")
    clip = homog @ V.T @ P.T
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None]
    return np.all(np.abs(ndc[:, :2]) < 1.0, axis=1) & (ndc[:, 2] > -1) & (ndc[:, 2] < 1) & (w > 0)

ROOT_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=6)
    ap.add_argument("--out", type=str, default="../data/pilot")
    ap.add_argument("--render", action="store_true", default=True)
    ap.add_argument("--no-render", dest="render", action="store_false")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    state_dir = os.path.join(out_dir, "state")
    os.makedirs(frames_dir, exist_ok=True)
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
            t0 = time.time()
            episode = run_episode(anchor, spec, client, render=args.render)
            dt = time.time() - t0

            # Observability gate: require the sphere to stay above the floor plane
            # (never fully below) and to be visible in-frame throughout scoring
            # (NDC-space check against the fixed camera, not a hardcoded radius).
            min_z = float(episode.position[:, 2].min())
            visible = bool(in_frame_mask(episode.position[episode.frame_steps]).all())
            has_contact = bool(episode.contact_flags.any())
            accepted = (min_z > -SPHERE_RADIUS_TOL) and visible

            clip_id = f"world{world_idx:03d}_{spec.name}"
            state_path = os.path.join(state_dir, f"{clip_id}.npz")
            np.savez_compressed(
                state_path,
                time_s=episode.time_s,
                position=episode.position,
                orientation=episode.orientation,
                linear_velocity=episode.linear_velocity,
                angular_velocity=episode.angular_velocity,
                contact_flags=episode.contact_flags,
                contact_impulses=episode.contact_impulses,
                frame_steps=episode.frame_steps,
            )

            frames_path = None
            if args.render and episode.frames is not None:
                frames_path = os.path.join(frames_dir, f"{clip_id}.npy")
                np.save(frames_path, episode.frames)

            manifest_rows.append({
                "clip_id": clip_id,
                "world_index": world_idx,
                "intervention": spec.name,
                "factor": spec.factor,
                "g": spec.g, "e": spec.e, "mu": spec.mu,
                "velocity_delta": spec.velocity_delta,
                "anchor_x0": anchor.x0, "anchor_y0": anchor.y0, "anchor_z0": anchor.z0,
                "anchor_heading": anchor.heading, "anchor_speed": anchor.horizontal_speed,
                "anchor_vz0": anchor.vz0,
                "accepted": accepted,
                "min_z_m": min_z,
                "has_contact": has_contact,
                "n_contact_steps": int(episode.contact_flags.sum()),
                "sim_seconds": dt,
                "state_path": os.path.relpath(state_path, out_dir),
                "frames_path": os.path.relpath(frames_path, out_dir) if frames_path else None,
            })
            if not accepted:
                rejected.append(clip_id)
            print(f"  [{clip_id}] contact_steps={int(episode.contact_flags.sum())} "
                  f"min_z={min_z:.3f} render_and_sim_time={dt:.2f}s")

    p.disconnect(client)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "root_seed": ROOT_SEED,
            "n_worlds": args.n_worlds,
            "n_clips": len(manifest_rows),
            "n_rejected": len(rejected),
            "rejected_clip_ids": rejected,
            "baseline": {"g": BASELINE_G, "e": BASELINE_E, "mu": BASELINE_MU},
            "wall_time_s": time.time() - t_start,
            "renderer": "pybullet_tinyrenderer_smoke_test" if args.render else None,
            "rows": manifest_rows,
        }, f, indent=2)

    print(f"\nWrote {len(manifest_rows)} clips ({len(rejected)} rejected) "
          f"in {time.time() - t_start:.1f}s -> {manifest_path}")


SPHERE_RADIUS_TOL = 0.16  # sphere radius (0.15) + small margin

if __name__ == "__main__":
    main()
