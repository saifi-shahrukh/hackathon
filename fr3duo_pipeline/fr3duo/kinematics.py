"""Damped-least-squares inverse kinematics for one FR3 arm on the composed RCS MuJoCo model.

Works on a private MjData so the simulation state is never touched. The target is the TCP
(attachment site + 0.1493 m along its z axis) position and a full orientation; the 7th degree of
freedom is pulled towards a posture reference (the home joints) in the null space.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .rcs_scene import TCP_OFFSET_M


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str, arm_qadr: np.ndarray, arm_dofadr: np.ndarray, site_id: int,
                 q_home: np.ndarray, limits: tuple[np.ndarray, np.ndarray] | None = None):
        self.m = model
        self.d = mujoco.MjData(model)
        self.side = side
        self.qadr, self.dadr, self.site = arm_qadr, arm_dofadr, site_id
        self.body = int(model.site_bodyid[site_id])
        self.q_home = np.asarray(q_home, dtype=float)
        if limits is None:
            lo = np.array([model.jnt_range[model.dof_jntid[i]][0] for i in arm_dofadr])
            hi = np.array([model.jnt_range[model.dof_jntid[i]][1] for i in arm_dofadr])
            margin = 0.02
            limits = (lo + margin, hi - margin)
        self.lo, self.hi = limits

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        rot = self.d.site_xmat[self.site].reshape(3, 3)
        return self.d.site_xpos[self.site] + TCP_OFFSET_M * rot[:, 2], rot

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray, q0: np.ndarray, *, iters: int = 60,
              pos_tol: float = 5e-4, rot_tol: float = 2e-3, damping: float = 1e-3, rot_weight: float = 0.5,
              null_gain: float = 0.05, step_limit: float = 0.2) -> tuple[np.ndarray, float, float]:
        """Returns (q, position error [m], orientation error [rad])."""
        q = np.array(q0, dtype=float)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            pos, rot = self.fk(q)
            e_pos = target_pos - pos
            # orientation error as rotation vector of R_target * R^T
            r_err = target_rot @ rot.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r_err.reshape(-1))
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jac(self.m, self.d, jacp, jacr, pos, self.body)
            J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
            e = np.concatenate([e_pos, rot_weight * e_rot])
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space pull towards the posture reference
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += null_gain * (N @ (self.q_home - q))
            n = np.linalg.norm(dq)
            if n > step_limit:
                dq *= step_limit / n
            q = np.clip(q + dq, self.lo, self.hi)
        pos, rot = self.fk(q)
        r_err = target_rot @ rot.T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, r_err.reshape(-1))
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, quat, 1.0)
        return q, float(np.linalg.norm(target_pos - pos)), float(np.linalg.norm(e_rot))


def rot_with_yaw(base_rot: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a TCP orientation about the world z axis by `yaw` (radians)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ base_rot
