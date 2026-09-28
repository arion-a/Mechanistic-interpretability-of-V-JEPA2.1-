"""Phase 2 of the parallelized Blender pipeline: one persistent Blender process
renders its shard of accepted clips from phase 1's saved state arrays.

Run via: blender --background --python blender_render_worker.py -- \
    --state-dir <dir> --out-dir <dir> --shard-id 0 --n-shards 16 [--image-size 384]

Reads the render result directly from Blender's pixel buffer (foreach_get into
a preallocated numpy array) instead of round-tripping through PNG files on
disk - the one-time ~34s GPU shader-compile cost is paid once per worker
process, then amortized across every clip that worker renders.
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import mathutils
import numpy as np

N_ENCODER_FRAMES = 64  # matches encode/encode_pilot.py's extraction contract

# Must stay in sync with sim/render.py's CAMERA_* constants (v3 calibration:
# bounded trajectory envelope after shrinking sample_anchor's speed range and
# adding ROLLING_FRICTION in world.py - see render.py's calibration history
# comment). Recompute via p.computeViewMatrixFromYawPitchRoll(...) + np.linalg.inv
# if those constants ever change; Blender's bundled Python has no pybullet to
# compute this matrix directly.
CAMERA_TO_WORLD = [
    [7.07106771e-01, -5.99660558e-01, 3.74709513e-01, 3.18503112e+00],
    [7.07106756e-01, 5.99660474e-01, -3.74709535e-01, -3.18503119e+00],
    [1.78712760e-08, 5.29919266e-01, 8.48048061e-01, 7.45840866e+00],
    [0.0, 0.0, 0.0, 1.0],
]
FOV_DEG = 34.0
SPHERE_RADIUS = 0.15


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--samples", type=int, default=32)
    return ap.parse_args(argv)


def setup_scene(image_size: int, samples: int):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 8.0
    checker.inputs["Color1"].default_value = (0.75, 0.82, 0.95, 1.0)
    checker.inputs["Color2"].default_value = (0.95, 0.95, 0.95, 1.0)
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    floor.data.materials.append(mat)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=(0, 0, SPHERE_RADIUS))
    sphere = bpy.context.active_object
    sphere.name = "Sphere"
    smat = bpy.data.materials.new("SphereMat")
    smat.use_nodes = True
    smat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1.0)
    sphere.data.materials.append(smat)
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.shade_smooth()

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.angle = math.radians(FOV_DEG)
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.matrix_world = mathutils.Matrix(CAMERA_TO_WORLD)
    scene.camera = cam_obj

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    # A grazing 55deg elevation cast a long, hard-edged shadow that read as a
    # second dark ball floating near the sphere (confirmed: disappears when
    # the sun is hidden from render). Lower elevation (steeper, more overhead
    # light) plus a wider sun angle (soft shadow) keeps the shadow small and
    # directly under the sphere instead.
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.rotation_euler = (math.radians(25), 0, math.radians(35))
    sun_data.angle = math.radians(5.0)
    scene.collection.objects.link(sun_obj)
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.image_settings.file_format = "PNG"
    scene.eevee.taa_render_samples = samples
    # Render to an in-memory Render Result rather than writing PNGs to disk.
    scene.render.filepath = "/tmp/blender_scratch_"

    return scene, sphere


def capture_frame(scratch_path: str, image_size: int) -> np.ndarray:
    """Render-to-file then read back via Blender's own image loader. The
    in-memory 'Render Result' pixel buffer (img.pixels.foreach_get) comes
    back empty in --background mode on this build, and Blender's bundled
    Python has neither PIL nor pip-installable packages available, so we
    round-trip through a scratch PNG using bpy.data.images.load() instead -
    deleting both the file and the loaded datablock immediately so nothing
    accumulates on disk or in memory."""
    bpy.context.scene.render.filepath = scratch_path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(scratch_path)
    buf = np.empty(image_size * image_size * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    os.remove(scratch_path)
    buf = np.flipud(buf.reshape((image_size, image_size, 4)))
    return np.clip(buf[:, :, :3] * 255.0, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    with open(os.path.join(args.state_dir, "..", "manifest.json")) as f:
        manifest = json.load(f)
    accepted = [r for r in manifest["rows"] if r["accepted"]]
    shard = [r for i, r in enumerate(accepted) if i % args.n_shards == args.shard_id]

    print(f"WORKER {args.shard_id} ASSIGNED {len(shard)} of {len(accepted)} accepted clips", flush=True)

    scene, sphere = setup_scene(args.image_size, args.samples)
    scratch_path = f"/tmp/blender_scratch_{args.shard_id}.png"

    t_start = time.time()
    for n, row in enumerate(shard):
        state_path = os.path.join(os.path.dirname(args.state_dir), row["state_path"])
        state = np.load(state_path)
        position, frame_steps_full = state["position"], state["frame_steps"]

        # Render only the 64 frames the encoder actually samples (uniform over
        # the full 180-frame timeline), not all 180 - cuts render work ~2.8x
        # and keeps per-clip disk footprint small regardless of dataset size.
        idx = np.round(np.linspace(0, len(frame_steps_full) - 1, N_ENCODER_FRAMES)).astype(int)
        frame_steps = frame_steps_full[idx]

        t0 = time.time()
        frames = np.empty((len(frame_steps), args.image_size, args.image_size, 3), dtype=np.uint8)
        for i, step in enumerate(frame_steps):
            pos = position[step]
            sphere.location = (float(pos[0]), float(pos[1]), float(pos[2]))
            frames[i] = capture_frame(scratch_path, args.image_size)
        dt = time.time() - t0

        out_path = os.path.join(frames_dir, f"{row['clip_id']}.npy")
        np.save(out_path, frames)
        print(f"WORKER {args.shard_id} CLIP_DONE {row['clip_id']} frames={len(frame_steps)} "
              f"time={dt:.2f}s ({n+1}/{len(shard)})", flush=True)

    total_dt = time.time() - t_start
    print(f"WORKER {args.shard_id} DONE clips={len(shard)} total_time={total_dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
