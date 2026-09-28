"""Encode-then-delete daemon: runs concurrently with the parallel Blender
render workers. Polls the frames directory, encodes any new clip through
V-JEPA the moment it appears, saves the 32,768-float feature vector, and
deletes the raw frames file immediately - so disk usage stays flat
(~feature-vector size only) regardless of how many clips the study has,
instead of needing to hold the full rendered dataset on disk at once.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_SPATIAL_PATCHES = (256 // 16) ** 2
NUM_TEMPORAL_TUBELETS = 64 // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def weights_hash(model) -> str:
    h = hashlib.sha256()
    for _, t in sorted(model.state_dict().items()):
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True, help="phase-1 manifest.json (for clip metadata)")
    ap.add_argument("--cache-dir", default="/workspace/hf_cache")
    ap.add_argument("--idle-timeout", type=float, default=120.0,
                     help="stop after this many seconds with no new files AND all accepted clips seen")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest) as f:
        manifest = json.load(f)
    by_clip_id = {r["clip_id"]: r for r in manifest["rows"] if r["accepted"]}
    total_expected = len(by_clip_id)

    processor = AutoVideoProcessor.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(MODEL_ID, cache_dir=args.cache_dir).to(DEVICE).eval()
    whash = weights_hash(model)
    print(f"SWEEPER device={DEVICE} weights_hash={whash} expecting {total_expected} clips", flush=True)

    done = set()
    rows = []
    t_start = time.time()
    last_progress = time.time()

    while len(done) < total_expected:
        pending = [f for f in os.listdir(args.frames_dir)
                   if f.endswith(".npy") and f[:-4] not in done]
        if not pending:
            if time.time() - last_progress > args.idle_timeout:
                print(f"SWEEPER idle timeout with {len(done)}/{total_expected} done, stopping", flush=True)
                break
            time.sleep(2.0)
            continue

        for fname in pending:
            clip_id = fname[:-4]
            path = os.path.join(args.frames_dir, fname)
            try:
                frames = np.load(path)  # [64, H, W, 3] uint8, already encoder-ready
            except Exception as e:
                continue  # file still being written by a render worker; retry next poll

            t0 = time.time()
            inputs = processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            tokens = out.last_hidden_state[0].cpu().view(NUM_TEMPORAL_TUBELETS, NUM_SPATIAL_PATCHES, -1)
            feature = tokens.mean(dim=1).flatten().numpy().astype(np.float32)
            dt = time.time() - t0

            feat_path = os.path.join(args.out_dir, f"{clip_id}.npy")
            np.save(feat_path, feature)
            os.remove(path)  # free disk immediately - the whole point of this daemon

            row = by_clip_id.get(clip_id, {})
            rows.append({
                "clip_id": clip_id, "world_index": row.get("world_index"),
                "intervention": row.get("intervention"), "factor": row.get("factor"),
                "feature_path": f"{clip_id}.npy", "feature_dim": int(feature.shape[0]),
                "encode_seconds": dt,
            })
            done.add(clip_id)
            last_progress = time.time()
            print(f"SWEEPER ENCODED {clip_id} ({len(done)}/{total_expected}) {dt:.2f}s", flush=True)

    meta = {
        "model_id": MODEL_ID, "device": DEVICE, "weights_hash": whash,
        "frames_per_clip": 64, "num_temporal_tubelets": NUM_TEMPORAL_TUBELETS,
        "num_spatial_patches": NUM_SPATIAL_PATCHES,
        "layer": "final_hidden_state", "pooling": "spatial_mean_per_tubelet_temporal_concat",
        "feature_dtype": "float32", "wall_time_s": time.time() - t_start,
        "rows": rows,
    }
    with open(os.path.join(args.out_dir, "features_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"SWEEPER_DONE encoded={len(done)} time={time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
