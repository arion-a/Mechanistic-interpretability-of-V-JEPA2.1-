"""Deterministic sphere-drop simulation with one-factor physics interventions.

Implements the scene and intervention spec from 01_3D_VJEPA_Research_Agenda.docx
section 02 ("World and interventions"): a 1 kg sphere of radius 0.15 m falling
above a horizontal plane, baseline g=9.8 / e=0.6 / mu=0.2, and eight one-factor
paired interventions (gravity, restitution, friction, initial velocity) plus
appearance-only interventions that never touch collision materials.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

SPHERE_RADIUS = 0.15
SPHERE_MASS = 1.0
PHYSICS_HZ = 240
SIM_SECONDS = 6.0
FPS = 30
N_FRAMES = int(SIM_SECONDS * FPS)  # 180
N_PHYSICS_STEPS = int(SIM_SECONDS * PHYSICS_HZ)  # 1440

BASELINE_G = 9.8
BASELINE_E = 0.6
BASELINE_MU = 0.2

# One-factor paired interventions, per the agenda doc.
PHYSICAL_FACTORS = {
    "gravity": {"low": 7.35, "high": 12.25},
    "restitution": {"low": 0.3, "high": 0.8},
    "friction": {"low": 0.1, "high": 0.4},
}
VELOCITY_DELTA = 0.15  # m/s, added along initial horizontal heading (deviation, see sample_anchor)
ROLLING_FRICTION = 0.02  # deviation from docx's rf=0: bounds worst-case excursion, applied
# identically across all conditions so it never confounds the sliding-friction (BASELINE_MU) intervention


def derive_seed(root_seed: int, role: str, index: int) -> int:
    """Deterministic 32-bit seed derivation (agenda: no hash(), no PID, no clock)."""
    payload = f"{root_seed}|v1|{role}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return struct.unpack(">I", digest[:4])[0]


@dataclass
class AnchorState:
    x0: float
    y0: float
    z0: float
    heading: float
    horizontal_speed: float
    vz0: float

    @property
    def vx0(self) -> float:
        return self.horizontal_speed * np.cos(self.heading)

    @property
    def vy0(self) -> float:
        return self.horizontal_speed * np.sin(self.heading)


def sample_anchor(root_seed: int, base_world_index: int) -> AnchorState:
    seed = derive_seed(root_seed, "sampler", base_world_index)
    rng = np.random.default_rng(seed)
    return AnchorState(
        x0=float(rng.uniform(-0.5, 0.5)),
        y0=float(rng.uniform(-0.5, 0.5)),
        z0=float(rng.uniform(1.5, 2.5)),
        heading=float(rng.uniform(0.0, 2 * np.pi)),
        # DEVIATION from docx spec (1.0-2.0 m/s): a fixed non-tracking camera
        # cannot both keep the full trajectory excursion in frame AND resolve
        # the sphere at a useful pixel size - measured real excursion at the
        # docx's original ranges reached 8-11m (with the velocity intervention),
        # leaving the sphere ~5px across (<1/3 of one ViT patch). Shrinking the
        # speed range bounds worst-case excursion to ~2.75m, sphere ~19px.
        # See experiment/remote/ for the PyBullet measurements behind this.
        horizontal_speed=float(rng.uniform(0.3, 0.6)),
        vz0=float(rng.uniform(-0.5, 0.5)),
    )


@dataclass
class InterventionSpec:
    name: str
    factor: str  # "gravity" | "restitution" | "friction" | "velocity" | "baseline"
    g: float = BASELINE_G
    e: float = BASELINE_E
    mu: float = BASELINE_MU
    velocity_delta: float = 0.0  # added along heading, m/s


def baseline_and_interventions() -> list[InterventionSpec]:
    specs = [InterventionSpec(name="baseline", factor="baseline")]
    for level_name, val in PHYSICAL_FACTORS["gravity"].items():
        specs.append(InterventionSpec(name=f"gravity_{level_name}", factor="gravity", g=val))
    for level_name, val in PHYSICAL_FACTORS["restitution"].items():
        specs.append(InterventionSpec(name=f"restitution_{level_name}", factor="restitution", e=val))
    for level_name, val in PHYSICAL_FACTORS["friction"].items():
        specs.append(InterventionSpec(name=f"friction_{level_name}", factor="friction", mu=val))
    specs.append(InterventionSpec(name="velocity_plus", factor="velocity", velocity_delta=+VELOCITY_DELTA))
    specs.append(InterventionSpec(name="velocity_minus", factor="velocity", velocity_delta=-VELOCITY_DELTA))
    return specs  # baseline + 8 interventions = 9, matches "nine videos per world"


@dataclass
class EpisodeResult:
    world_index: int
    intervention: str
    factor: str
    anchor: AnchorState
    time_s: np.ndarray          # [T]
    position: np.ndarray        # [T,3]
    orientation: np.ndarray     # [T,4] quaternion
    linear_velocity: np.ndarray  # [T,3]
    angular_velocity: np.ndarray  # [T,3]
    contact_flags: np.ndarray   # [T] bool, any contact this physics sub-step
    contact_impulses: np.ndarray  # [T] float, summed normal impulse magnitude
    frame_steps: np.ndarray     # [N_FRAMES] indices into the physics-step arrays above
    frames: Optional[np.ndarray] = field(default=None)  # [N_FRAMES, H, W, 3] uint8, if rendered


def run_episode(anchor: AnchorState, spec: InterventionSpec, physics_client: int,
                 render: bool = False) -> EpisodeResult:
    """Deterministic fixed-step rigid-body simulation of one baseline/intervention clip."""
    p.resetSimulation(physicsClientId=physics_client)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0 / PHYSICS_HZ, numSolverIterations=50,
                                 physicsClientId=physics_client)
    p.setGravity(0, 0, -spec.g, physicsClientId=physics_client)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client)
    p.changeDynamics(plane_id, -1, restitution=spec.e, lateralFriction=spec.mu,
                      rollingFriction=ROLLING_FRICTION, spinningFriction=ROLLING_FRICTION,
                      physicsClientId=physics_client)

    col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS, physicsClientId=physics_client)
    vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS, rgbaColor=[0.8, 0.2, 0.2, 1.0],
                                     physicsClientId=physics_client)
    body_id = p.createMultiBody(
        baseMass=SPHERE_MASS,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=[anchor.x0, anchor.y0, anchor.z0],
        physicsClientId=physics_client,
    )
    p.changeDynamics(body_id, -1, restitution=spec.e, lateralFriction=spec.mu,
                      rollingFriction=ROLLING_FRICTION, spinningFriction=ROLLING_FRICTION,
                      linearDamping=0.0, angularDamping=0.0, physicsClientId=physics_client)

    vx0 = anchor.vx0 + spec.velocity_delta * np.cos(anchor.heading)
    vy0 = anchor.vy0 + spec.velocity_delta * np.sin(anchor.heading)
    p.resetBaseVelocity(body_id, linearVelocity=[vx0, vy0, anchor.vz0], angularVelocity=[0, 0, 0],
                         physicsClientId=physics_client)

    T = N_PHYSICS_STEPS + 1
    time_s = np.zeros(T)
    position = np.zeros((T, 3))
    orientation = np.zeros((T, 4))
    linear_velocity = np.zeros((T, 3))
    angular_velocity = np.zeros((T, 3))
    contact_flags = np.zeros(T, dtype=bool)
    contact_impulses = np.zeros(T)

    def record(i):
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
        lin, ang = p.getBaseVelocity(body_id, physicsClientId=physics_client)
        position[i] = pos
        orientation[i] = orn
        linear_velocity[i] = lin
        angular_velocity[i] = ang
        time_s[i] = i / PHYSICS_HZ

    frame_steps = np.round(np.linspace(0, N_PHYSICS_STEPS, N_FRAMES)).astype(int)
    frame_set = set(frame_steps.tolist())
    frames = np.zeros((N_FRAMES, 1, 1, 3), dtype=np.uint8)  # placeholder shape, replaced below if render
    collected_frames = []

    if render:
        import render as render_mod  # local import: avoid pybullet camera setup cost when unused

    record(0)
    if render and 0 in frame_set:
        collected_frames.append((0, render_mod.capture_frame(physics_client)))

    for i in range(1, T):
        p.stepSimulation(physicsClientId=physics_client)
        contacts = p.getContactPoints(bodyA=body_id, bodyB=plane_id, physicsClientId=physics_client)
        if contacts:
            contact_flags[i] = True
            contact_impulses[i] = sum(c[9] for c in contacts)  # normalForce index
        record(i)
        if render and i in frame_set:
            collected_frames.append((i, render_mod.capture_frame(physics_client)))

    if render:
        collected_frames.sort(key=lambda t: t[0])
        frames = np.stack([f for _, f in collected_frames], axis=0)
    else:
        frames = None

    return EpisodeResult(
        world_index=-1,
        intervention=spec.name,
        factor=spec.factor,
        anchor=anchor,
        time_s=time_s,
        position=position,
        orientation=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        contact_flags=contact_flags,
        contact_impulses=contact_impulses,
        frame_steps=frame_steps,
        frames=frames,
    )
