"""Encode pilot clips through V-JEPA 2 and save the primary representation.

Representation contract (01_3D_VJEPA_Research_Agenda.docx, "Extraction contract"):
final encoder layer, mean over spatial tokens within each temporal tubelet,
then concatenate temporal vectors. 64 uniformly spaced frames are the primary
input, subsampled from the full 6s / 180-frame rendered clip.

Model note: facebook/vjepa2-vitl-fpc64-256 is a documented substitution for
the docx's unpublished V-JEPA 2.1 ViT-B/16 384px checkpoint (see M0 report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
FRAMES_PER_CLIP = 64  # model's native fpc; also the docx's primary-input frame count
NUM_SPATIAL_PATCHES = (256 // 16) ** 2  # 256
NUM_TEMPORAL_TUBELETS = FRAMES_PER_CLIP // 2  # tubelet_size=2 -> 32


def weights_hash(model) -> str:
    h = hashlib.sha256()
    for _, t in sorted(model.state_dict().items()):
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def subsample_64(frames: np.ndarray) -> np.ndarray:
    """Uniformly select 64 frames spanning the full clip, matching the extraction contract."""
    n = frames.shape[0]
    idx = np.round(np.linspace(0, n - 1, FRAMES_PER_CLIP)).astype(int)
    return frames[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default="../data/pilot")
    ap.add_argument("--cache-dir", default="../hf_cache")
    ap.add_argument("--out", default="../data/pilot/features")
    ap.add_argument("--limit", type=int, default=None, help="cap number of clips (debug)")
    args = ap.parse_args()

    pilot_dir = os.path.abspath(args.pilot_dir)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(pilot_dir, "manifest.json")) as f:
        manifest = json.load(f)

    rows = [r for r in manifest["rows"] if r["accepted"] and r["frames_path"]]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Encoding {len(rows)} accepted clips with {MODEL_ID}")

    processor = AutoVideoProcessor.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 4)
    whash = weights_hash(model)
    print("weights_hash:", whash)

    feature_rows = []
    t_start = time.time()
    for i, row in enumerate(rows):
        clip_id = row["clip_id"]
        frames_path = os.path.join(pilot_dir, row["frames_path"])
        frames = np.load(frames_path)  # [180, 256, 256, 3] uint8
        clip64 = subsample_64(frames)  # [64, 256, 256, 3]

        inputs = processor(list(clip64), return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = model(**inputs)
        dt = time.time() - t0

        tokens = out.last_hidden_state[0]  # [8192, 1024]
        tokens = tokens.view(NUM_TEMPORAL_TUBELETS, NUM_SPATIAL_PATCHES, -1)  # [32, 256, 1024]
        temporal_vecs = tokens.mean(dim=1)  # [32, 1024], spatial-mean per tubelet
        feature = temporal_vecs.flatten().numpy().astype(np.float32)  # [32768]

        feat_path = os.path.join(out_dir, f"{clip_id}.npy")
        np.save(feat_path, feature)

        feature_rows.append({
            "clip_id": clip_id,
            "world_index": row["world_index"],
            "intervention": row["intervention"],
            "factor": row["factor"],
            "feature_path": os.path.relpath(feat_path, out_dir),
            "feature_dim": int(feature.shape[0]),
            "encode_seconds": dt,
        })
        print(f"  [{i+1}/{len(rows)}] {clip_id} -> dim={feature.shape[0]} ({dt:.1f}s)")

    meta = {
        "model_id": MODEL_ID,
        "weights_hash": whash,
        "frames_per_clip": FRAMES_PER_CLIP,
        "num_temporal_tubelets": NUM_TEMPORAL_TUBELETS,
        "num_spatial_patches": NUM_SPATIAL_PATCHES,
        "layer": "final_hidden_state",
        "pooling": "spatial_mean_per_tubelet_temporal_concat",
        "feature_dtype": "float32",
        "wall_time_s": time.time() - t_start,
        "rows": feature_rows,
    }
    with open(os.path.join(out_dir, "features_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nEncoded {len(feature_rows)} clips in {time.time()-t_start:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
