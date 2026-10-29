"""Thin helpers around robot-control-stack's FR3 Duo MuJoCo scene (RCS venv only).

The organizer dataset `fr3_duo_rand500` was rendered with this exact scene (same home pose to
1e-6 rad, same three camera placements), so everything here builds on `EmptyWorldFR3Duo` and never
edits the RCS repository.

Conventions verified on 2026-10-29 (fr3duo_pipeline/tools/calibrate_scene.py prints them again):
- MuJoCo qpos layout: [left joints 0..6, left gripper 6 joints, right joints 0..6, right gripper 6].
- TCP (2F-85 pad centre) = `robot<side>_attachment_site` position + 0.1493 m along the site z axis.
- RCS gripper observation = 1 - driver joint angle [rad]; 1.0 = open. Dataset convention is the
  same (1 = open, binary command, continuous state).
- `set_joints_hard` writes qpos and the actuator targets but does NOT refresh the RCS-side cached
  Cartesian pose; read kinematics from MuJoCo after `mj_forward`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

TCP_OFFSET_M = 0.1493  # rcs.GRIPPER_TCP_OFFSETS[Robotiq2F85].translation()[2]
SIDES = ("left", "right")
Q_SLICES = {"left": slice(0, 7), "right": slice(8, 15)}
G_INDEX = {"left": 7, "right": 15}


def make_duo_env(*, headless: bool = True, frequency: float = 6.0, cam_res: int = 224,
                 world_frame_objects: dict | None = None, async_control: bool = True):
    """EmptyWorldFR3Duo with absolute joint control at `frequency` Hz and square cameras."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rcs.envs.base import ControlMode, RelativeTo
    from rcs.envs.configs import EmptyWorldFR3Duo

    scene = EmptyWorldFR3Duo()
    cfg = scene.config()
    cfg.headless = headless
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.NONE
    cfg.max_relative_movement = None
    cfg.sim_cfg.realtime = False
    cfg.sim_cfg.async_control = async_control
    cfg.sim_cfg.frequency = frequency
    for c in cfg.camera_cfgs.values():
        c.resolution_width = cam_res
        c.resolution_height = cam_res
    if world_frame_objects:
        cfg.world_frame_objects = {**(cfg.world_frame_objects or {}), **world_frame_objects}
    env = scene.create_env(cfg)
    return env, cfg


@dataclass
class DuoHandles:
    env: object
    sim: object
    model: mujoco.MjModel
    data: mujoco.MjData
    robots: dict
    grippers: dict
    site_id: dict
    arm_qadr: dict      # side -> 7 qpos addresses
    arm_dofadr: dict    # side -> 7 qvel addresses
    arm_actadr: dict    # side -> 7 actuator ids
    grip_actadr: dict   # side -> gripper actuator id
    driver_qadr: dict   # side -> qpos address of the gripper driver joint

    @classmethod
    def from_env(cls, env) -> "DuoHandles":
        sim = env.get_wrapper_attr("sim")
        m, d = sim.model, sim.data
        jid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)  # noqa: E731
        site_id = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"robot{s}_attachment_site") for s in SIDES}
        arm_qadr = {s: np.array([m.jnt_qposadr[jid(f"robot{s}_fr3_joint{i}")] for i in range(1, 8)]) for s in SIDES}
        arm_dofadr = {s: np.array([m.jnt_dofadr[jid(f"robot{s}_fr3_joint{i}")] for i in range(1, 8)]) for s in SIDES}
        arm_actadr = {s: np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot{s}_fr3_joint{i}")
                                   for i in range(1, 8)]) for s in SIDES}
        grip_actadr = {s: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"gripper{s}_fingers_actuator")) for s in SIDES}
        driver_qadr = {s: int(m.jnt_qposadr[jid(f"gripper{s}_right_driver_joint")]) for s in SIDES}
        return cls(env, sim, m, d, env.get_wrapper_attr("robot"), env.get_wrapper_attr("gripper"), site_id,
                   arm_qadr, arm_dofadr, arm_actadr, grip_actadr, driver_qadr)

    def set_arms_hard(self, state16: np.ndarray, sync_ctrl: bool = True) -> None:
        """Write both arms' joints (and the gripper driver joints from the normalized state) into qpos.

        With sync_ctrl the position actuators' targets follow, so stepping physics afterwards holds
        the pose instead of pulling the arm back to the previous target. Gripper ctrl is 0..255
        (0 = open) in RCS's 2F-85 model: ctrl = (1 - normalized) * 255."""
        for s in SIDES:
            self.data.qpos[self.arm_qadr[s]] = state16[Q_SLICES[s]]
            self.data.qvel[self.arm_dofadr[s]] = 0.0
            self.data.qpos[self.driver_qadr[s]] = float(np.clip(1.0 - state16[G_INDEX[s]], 0.0, 0.9))
            if sync_ctrl:
                self.data.ctrl[self.arm_actadr[s]] = state16[Q_SLICES[s]]
                self.data.ctrl[self.grip_actadr[s]] = float(np.clip(1.0 - state16[G_INDEX[s]], 0.0, 1.0)) * 255.0
        mujoco.mj_forward(self.model, self.data)

    def tcp_world(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(xyz, 3x3 rotation) of the TCP in world coordinates, from the current MuJoCo state."""
        sid = self.site_id[side]
        rot = self.data.site_xmat[sid].reshape(3, 3)
        return self.data.site_xpos[sid] + TCP_OFFSET_M * rot[:, 2], rot

    def arm_joints(self, side: str) -> np.ndarray:
        return np.array(self.data.qpos[self.arm_qadr[side]])

    def gripper_obs(self, side: str) -> float:
        return float(self.grippers[side].get_normalized_width())


def action_dict(action16: np.ndarray) -> dict:
    """16-D dataset action -> RCS duo env action (absolute joints + gripper 0..1)."""
    return {s: {"joints": np.asarray(action16[Q_SLICES[s]], dtype=np.float64),
                "gripper": np.array([float(action16[G_INDEX[s]])], dtype=np.float32)} for s in SIDES}


def obs_to_state16(obs: dict) -> np.ndarray:
    out = np.zeros(16, dtype=np.float32)
    for s in SIDES:
        out[Q_SLICES[s]] = obs[s]["joints"]
        out[G_INDEX[s]] = float(np.asarray(obs[s]["gripper"]).reshape(-1)[0])
    return out


def yaw_of(rot: np.ndarray) -> float:
    """Yaw (about world z) of the gripper's closing axis (its y axis) modulo pi, in radians."""
    y = rot[:, 1]
    return float(np.arctan2(y[1], y[0]) % np.pi)
