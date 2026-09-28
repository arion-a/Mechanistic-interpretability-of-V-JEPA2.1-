"""Fixed-camera frame rendering for episodes.

Renders with PyBullet's built-in TinyRenderer. The research agenda doc
(01_3D_VJEPA_Research_Agenda.docx, "Recommended stack") specifies Blender
EEVEE for every final confirmatory split and reserves TinyRenderer explicitly
for "engineering smoke tests" — this module implements that smoke-test path.
Swap in a BlenderRenderer with the same call signature for confirmatory runs.
"""
from __future__ import annotations

import numpy as np
import pybullet as p

IMAGE_SIZE = 256  # matches the V-JEPA 2 ViT-L/16-256 checkpoint's crop_size

# Calibration history (see M1 observability gate in the agenda doc):
# v1 (azimuth 45, elevation 25, ~4.5m) kept only ~59% of frames in view with
# rollingFriction=0 letting the sphere roll indefinitely once past sliding.
# v2 widened the camera to cover the full ~8m excursion, which technically
# passed the in-frame check but made the sphere only ~5px across (<1/3 of one
# ViT patch) - resolvable-in-NDC is not the same as resolvable-in-detail, and
# that gap wasn't caught until real rendered output was inspected. v3 (this
# one) instead bounds the excursion itself: sample_anchor's speed range was
# shrunk and a small ROLLING_FRICTION added (see world.py), cutting worst-case
# excursion from ~8-11m to ~2.75m. This camera is calibrated to that bounded
# envelope (100% in-frame across 60 worlds x 9 clips x 180 frames), giving a
# sphere ~23px across - a real, not just nominal, improvement.
CAMERA_AZIMUTH_DEG = 45.0
CAMERA_ELEVATION_DEG = -58.0  # pybullet elevation is negative-down convention
CAMERA_DISTANCE = 8.5
CAMERA_TARGET = [0.0, 0.0, 0.25]
CAMERA_FOV_DEG = 34.0


_VIEW_MATRIX = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=CAMERA_TARGET,
    distance=CAMERA_DISTANCE,
    yaw=CAMERA_AZIMUTH_DEG,
    pitch=CAMERA_ELEVATION_DEG,
    roll=0,
    upAxisIndex=2,
)
_PROJ_MATRIX = p.computeProjectionMatrixFOV(fov=CAMERA_FOV_DEG, aspect=1.0, nearVal=0.1, farVal=30.0)


def capture_frame(physics_client: int) -> np.ndarray:
    """Render the current scene state with the fixed oblique camera."""
    _, _, rgba, _, _ = p.getCameraImage(
        width=IMAGE_SIZE, height=IMAGE_SIZE,
        viewMatrix=_VIEW_MATRIX, projectionMatrix=_PROJ_MATRIX,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=physics_client,
    )
    rgb = np.reshape(rgba, (IMAGE_SIZE, IMAGE_SIZE, 4))[:, :, :3].astype(np.uint8)
    return rgb
