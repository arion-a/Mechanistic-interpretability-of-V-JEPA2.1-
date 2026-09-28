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

# Calibration note (pilot finding, see M1 observability gate in the agenda doc):
# the doc's nominal camera (azimuth 45, elevation 25, aimed at floor center,
# implied distance ~4.5m) keeps only ~59% of frames in view. With
# rollingFriction=0 (as specified) the sphere never stops rolling and its
# horizontal displacement over the full 6s clip reaches several meters, well
# beyond a tight framing. Recalibrated empirically against this pilot's own
# trajectory envelope (see sim/calibrate_camera.py) to keep 100% of sampled
# frames in-frame while preserving the specified 45-degree azimuth.
CAMERA_AZIMUTH_DEG = 45.0
CAMERA_ELEVATION_DEG = -45.0  # pybullet elevation is negative-down convention
CAMERA_DISTANCE = 16.0
CAMERA_TARGET = [-3.5, -1.8, 0.4]
CAMERA_FOV_DEG = 85.0


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
