"""Blender EEVEE render benchmark, driven by world000_baseline's real trajectory.

Run headless on the pod: blender --background --python blender_bench.py
"""
import bpy
import numpy as np
import time
import math
import mathutils

STATE_PATH = "/workspace/vjepa_experiment/world000_baseline_state.npz"
OUT_DIR = "/workspace/vjepa_experiment/blender_frames"
IMAGE_SIZE = 384  # docx's confirmatory-study target resolution
N_SAMPLE_FRAMES = 5

CAMERA_TO_WORLD = [
    [7.07106741e-01, -5.00000092e-01, 5.00000007e-01, 4.49999996e+00],
    [7.07106846e-01, 5.00000007e-01, -4.99999923e-01, -9.79999944e+00],
    [1.49011584e-08, 7.07106762e-01, 7.07106825e-01, 1.17137083e+01],
    [0.0, 0.0, 0.0, 1.0],
]
FOV_DEG = 85.0
SPHERE_RADIUS = 0.15

import os
os.makedirs(OUT_DIR, exist_ok=True)

# --- clear default scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- floor: large plane with a checker material ---
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "Floor"
mat = bpy.data.materials.new("FloorMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
bsdf = nodes["Principled BSDF"]
checker = nodes.new("ShaderNodeTexChecker")
checker.inputs["Scale"].default_value = 8.0
checker.inputs["Color1"].default_value = (0.75, 0.82, 0.95, 1.0)
checker.inputs["Color2"].default_value = (0.95, 0.95, 0.95, 1.0)
links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
floor.data.materials.append(mat)

# --- sphere ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=(0, 0, SPHERE_RADIUS))
sphere = bpy.context.active_object
sphere.name = "Sphere"
smat = bpy.data.materials.new("SphereMat")
smat.use_nodes = True
smat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1.0)
sphere.data.materials.append(smat)
bpy.ops.object.shade_smooth()

# --- camera, matching the exact pilot camera pose ---
cam_data = bpy.data.cameras.new("Cam")
cam_data.angle = math.radians(FOV_DEG)
cam_data.sensor_fit = "AUTO"
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.matrix_world = mathutils.Matrix(CAMERA_TO_WORLD)
scene.camera = cam_obj

# --- lighting ---
sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 3.0
sun_obj = bpy.data.objects.new("Sun", sun_data)
sun_obj.rotation_euler = (math.radians(55), 0, math.radians(35))
scene.collection.objects.link(sun_obj)
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4

# --- render settings: EEVEE, matching model input resolution target ---
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = IMAGE_SIZE
scene.render.resolution_y = IMAGE_SIZE
scene.render.image_settings.file_format = "PNG"
scene.eevee.taa_render_samples = 32

# --- drive sphere with the real saved trajectory ---
state = np.load(STATE_PATH)
position = state["position"]
frame_steps = state["frame_steps"]
idx = np.round(np.linspace(0, len(frame_steps) - 1, N_SAMPLE_FRAMES)).astype(int)

timings = []
for k, i in enumerate(idx):
    step = frame_steps[i]
    pos = position[step]
    sphere.location = (float(pos[0]), float(pos[1]), float(pos[2]))
    scene.render.filepath = f"{OUT_DIR}/frame_{k:02d}.png"
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    timings.append(dt)
    print(f"RENDER_FRAME {k} step={step} pos={pos.tolist()} time={dt:.4f}s", flush=True)

print(f"TIMINGS {timings}", flush=True)
print(f"MEAN_TIME {sum(timings)/len(timings):.4f}", flush=True)
print("BLENDER_BENCH_DONE", flush=True)
