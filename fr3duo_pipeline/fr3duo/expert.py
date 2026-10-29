"""Scripted bimanual cube-stacking expert for the CubeScene (RCS venv only).

Mirrors the structure measured in fr3_duo_rand500 (results/fr3_duo_rand500_audit.json):
the LEFT arm picks cube B (the larger-y one of the two non-base cubes) and places it on the base
cube A (gripper closes at ~frame 43, opens at ~109), then the RIGHT arm picks cube C and places it
on top (closes ~215, opens ~288); ~350 frames at 6 fps. Grasp height, release clearances and the
TCP orientation (the home orientation: ~20 deg pitch, closing axis along world y) come from the
calibration of the organizer data (configs/scene_calib.json). Cube poses are read from the
simulator (ground truth), as a scripted generator would. Actions are absolute joint targets per
frame plus a binary gripper command (1 = open, 0 = close), exactly the dataset's convention.

The hackathon's own expert was deleted; this is a re-implementation from the data's statistics,
not a copy. Timing jitter per episode comes from the seed so the 200 native episodes are not
clones of one schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import ArmIK
from .rcs_scene import G_INDEX, Q_SLICES, SIDES


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


@dataclass
class ArmPlan:
    """Piecewise linear Cartesian plan: list of (end_frame, target_pos or None=hold, gripper)."""
    segments: list   # (f_end, pos(3) | None, gripper_cmd)
    start_frame: int


class ScriptedStacker:
    GRASP_ABOVE_TABLE = 0.0374    # calibrated grasp TCP height above the table surface (cube centre at 0.021)
    HOVER_Z = 0.17
    PLACE_CLEARANCE = (0.009, 0.004)   # left (B on A), right (C on B): release heights above "on top"

    def __init__(self, scene, rng: np.random.Generator | None = None):
        self.sc = scene
        self.rng = rng or np.random.default_rng(0)
        # table surface height in world coordinates (0 for the flat scene, the tray floor in the clinic scene)
        self.GRASP_TCP_Z = float(getattr(scene, "table_z_world", 0.0)) + self.GRASP_ABOVE_TABLE
        h = scene.h
        self.ik = {}
        self.q_home = {}
        self.rot_home = {}
        for s in SIDES:
            q = h.arm_joints(s)
            self.q_home[s] = q.copy()
            self.ik[s] = ArmIK(h.model, s, h.arm_qadr[s], h.arm_dofadr[s], h.site_id[s], q)
            self.rot_home[s] = self.ik[s].fk(q)[1].copy()
        self.plans: dict[str, ArmPlan] = {}
        self.q_cmd = {s: self.q_home[s].copy() for s in SIDES}
        self.grip_cmd = {s: 1.0 for s in SIDES}
        self.seg_start_pos = {}
        self.seg_start_frame = {}
        self.ik_err_log = []

    # ── planning ────────────────────────────────────────────────────────────────────────────────
    def _jit(self, base: int, spread: int = 3) -> int:
        return int(base + self.rng.integers(-spread, spread + 1))

    def plan_left(self, pos_b: np.ndarray, pos_a: np.ndarray) -> ArmPlan:
        h2 = 2 * self.sc.p.cube_half
        j = self._jit
        grasp = np.array([pos_b[0], pos_b[1], self.GRASP_TCP_Z])
        place = np.array([pos_a[0], pos_a[1], self.GRASP_TCP_Z + h2 + self.PLACE_CLEARANCE[0]])
        segs = [
            (j(30), grasp + [0, 0, self.HOVER_Z], 1.0),     # hover above B
            (j(40), grasp, 1.0),                             # descend
            (j(43, 1), grasp, 1.0),                          # settle
            (j(49, 1), grasp, 0.0),                          # close + dwell
            (j(62), grasp + [0, 0, self.HOVER_Z], 0.0),     # lift
            (j(90), place + [0, 0, self.HOVER_Z - h2], 0.0),  # carry above A
            (j(106), place, 0.0),                            # descend onto A
            (j(109, 1), place, 0.0),
            (j(114, 1), place, 1.0),                         # open + dwell
            (j(126), place + [0, 0, self.HOVER_Z - h2], 1.0),  # lift
            (j(160), None, 1.0),                             # home
        ]
        return ArmPlan(segs, 0)

    def plan_right(self, pos_c: np.ndarray, pos_b_now: np.ndarray, t0: int) -> ArmPlan:
        h2 = 2 * self.sc.p.cube_half
        j = self._jit  # segment ends are relative to start_frame = t0
        grasp = np.array([pos_c[0], pos_c[1], self.GRASP_TCP_Z])
        place = np.array([pos_b_now[0], pos_b_now[1], self.GRASP_TCP_Z + 2 * h2 + self.PLACE_CLEARANCE[1]])
        segs = [
            (j(30), grasp + [0, 0, self.HOVER_Z], 1.0),
            (j(40), grasp, 1.0),
            (j(43, 1), grasp, 1.0),
            (j(49, 1), grasp, 0.0),
            (j(62), grasp + [0, 0, self.HOVER_Z], 0.0),
            (j(90), place + [0, 0, self.HOVER_Z - 2 * h2], 0.0),
            (j(110), place, 0.0),
            (j(113, 1), place, 0.0),
            (j(118, 1), place, 1.0),
            (j(130), place + [0, 0, self.HOVER_Z - 2 * h2], 1.0),
            (j(165), None, 1.0),
        ]
        return ArmPlan(segs, t0)

    def reset(self) -> None:
        pos = self.sc.cube_pos()
        self.plans = {"left": self.plan_left(pos["cube_b"], pos["cube_a"])}
        self.right_t0 = self._jit(172, 4)
        self.q_cmd = {s: self.q_home[s].copy() for s in SIDES}
        self.grip_cmd = {s: 1.0 for s in SIDES}
        self.seg_start_pos = {}
        self.seg_start_frame = {}
        self.ik_err_log = []
        self.total_frames = self.right_t0 + 165 + int(self.rng.integers(8, 16))

    # ── per-frame action ────────────────────────────────────────────────────────────────────────
    def _arm_action(self, side: str, frame: int) -> tuple[np.ndarray, float]:
        plan = self.plans.get(side)
        if plan is None or frame < plan.start_frame:
            return self.q_home[side], 1.0
        f = frame - plan.start_frame
        prev_end = 0
        for si, (f_end, target, grip) in enumerate(plan.segments):
            if f <= f_end:
                key = (side, si)
                if key not in self.seg_start_pos:
                    self.seg_start_pos[key] = (self.ik[side].fk(self.q_cmd[side])[0].copy(), self.q_cmd[side].copy())
                    self.seg_start_frame[key] = prev_end
                p0, q0 = self.seg_start_pos[key]
                dur = max(1, f_end - self.seg_start_frame[key])
                t = smoothstep((f - self.seg_start_frame[key]) / dur)
                if target is None:
                    # smooth joint-space return home
                    q = q0 + (self.q_home[side] - q0) * t
                    self.q_cmd[side] = q
                    return q, grip
                p = p0 + (np.asarray(target) - p0) * t
                q, e_pos, e_rot = self.ik[side].solve(p, self.rot_home[side], self.q_cmd[side], iters=25)
                self.ik_err_log.append(e_pos)
                self.q_cmd[side] = q
                return q, grip
            prev_end = f_end
        return self.q_home[side], 1.0

    def act(self, frame: int) -> np.ndarray:
        if "right" not in self.plans and frame >= self.right_t0 - 1:
            pos = self.sc.cube_pos()
            self.plans["right"] = self.plan_right(pos["cube_c"], pos["cube_b"], self.right_t0)
        a = np.zeros(16, dtype=np.float32)
        for s in SIDES:
            q, g = self._arm_action(s, frame)
            a[Q_SLICES[s]] = q
            a[G_INDEX[s]] = g
        return a

    def done(self, frame: int) -> bool:
        return frame >= self.total_frames
