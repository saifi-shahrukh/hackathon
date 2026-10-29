"""Reconstruct the cube-stacking scene geometry of fr3_duo_rand500 from forward kinematics.

    MUJOCO_GL=egl <rcs venv>/bin/python fr3duo_pipeline/tools/calibrate_scene.py \
        results/01_dataset_audit/rand500_frames.npz --out fr3duo_pipeline/configs/scene_calib.json

No cube poses are stored in the dataset, but every episode grasps and releases with a fixed
script (left: close at ~43, open at ~109; right: close at ~215, open at ~288). The TCP pose at
those frames tells where the cubes were:
  cube B initial  = left  TCP at its close frame      (left picks B)
  cube A          = left  TCP at its open frame       (B is placed on A; A never moves)
  cube C initial  = right TCP at its close frame      (right picks C)
  right open      = should coincide with A/B (C placed on top of the B-on-A stack)
The vertical differences give the cube height h: left open z - left close z = h,
right open z - right close z = 2h. Table top z = median grasp z - h/2.
All numbers are measured from the composed RCS MuJoCo model (same kinematics as the dataset's
generator) and written with their spread so the scene randomisation reproduces the data's ranges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fr3duo_pipeline.fr3duo.rcs_scene import DuoHandles, make_duo_env, yaw_of  # noqa: E402


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "mean": float(x.mean()), "std": float(x.std()),
            "p01": float(np.percentile(x, 1)), "p99": float(np.percentile(x, 99)),
            "min": float(x.min()), "max": float(x.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=Path("fr3duo_pipeline/configs/scene_calib.json"))
    ap.add_argument("--settle", type=int, default=3, help="frames after the close command at which the grasp pose is read")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    z = np.load(a.npz)
    st, ev, ep_start, ep_len = z["state"], z["events"], z["ep_start"], z["ep_len"]
    n = len(ep_start) if a.episodes == 0 else min(a.episodes, len(ep_start))

    env, _ = make_duo_env(headless=True, cam_res=64)
    env.reset()
    h = DuoHandles.from_env(env)
    env.reset()
    # verify the TCP convention at home against RCS's own Cartesian reading
    h.set_arms_hard(st[ep_start[0]])
    rcs_tcp = h.robots["left"].to_pose_in_world_coordinates(h.robots["left"].get_cartesian_position()).translation()
    mj_tcp, _ = h.tcp_world("left")
    tcp_check = float(np.linalg.norm(rcs_tcp - mj_tcp))
    print(f"[calib] TCP convention check at home: |rcs - mujoco| = {tcp_check * 1000:.2f} mm")

    rec = {k: [] for k in ("L_close", "L_open", "R_close", "R_open")}
    yaw = {k: [] for k in rec}
    zmin = {"left": [], "right": []}
    pitch = {k: [] for k in rec}
    for i in range(n):
        s0 = ep_start[i]
        frames = {"L_close": ev[i, 0] + a.settle, "L_open": ev[i, 1], "R_close": ev[i, 2] + a.settle, "R_open": ev[i, 3]}
        for k, f in frames.items():
            side = "left" if k[0] == "L" else "right"
            h.set_arms_hard(st[s0 + min(f, ep_len[i] - 1)])
            p, rot = h.tcp_world(side)
            rec[k].append(p.copy())
            yaw[k].append(yaw_of(rot))
            pitch[k].append(float(np.degrees(np.arccos(np.clip(-rot[2, 2], -1, 1)))))  # 0 = pointing straight down
        # lowest TCP z of each arm during its carry phase (gripper closed): grasp/placement heights
        for side, (c, o) in (("left", (ev[i, 0], ev[i, 1])), ("right", (ev[i, 2], ev[i, 3]))):
            zs = []
            for f in range(c, o + 1, 4):
                h.set_arms_hard(st[s0 + f])
                zs.append(h.tcp_world(side)[0][2])
            zmin[side].append(min(zs))
        if i % 50 == 0:
            print(f"[calib] episode {i}/{n}")
    rec = {k: np.stack(v) for k, v in rec.items()}

    h_left = rec["L_open"][:, 2] - rec["L_close"][:, 2]
    h_right = (rec["R_open"][:, 2] - rec["R_close"][:, 2]) / 2.0
    cube_h = float(np.median(np.concatenate([h_left, h_right])))
    grasp_z = np.concatenate([rec["L_close"][:, 2], rec["R_close"][:, 2]])
    table_z = float(np.median(grasp_z) - cube_h / 2)
    a_minus_ropen = np.linalg.norm(rec["L_open"][:, :2] - rec["R_open"][:, :2], axis=1)

    out = {
        "source": str(a.npz), "episodes": int(n), "settle_frames_after_close": a.settle,
        "tcp_convention_check_mm": tcp_check * 1000,
        "cube_height_m": {"median_combined": cube_h, "from_left_open_minus_close": stats(h_left),
                          "from_right_half_open_minus_close": stats(h_right)},
        "table_top_z_m": {"estimate": table_z, "grasp_center_z": stats(grasp_z),
                          "lowest_tcp_z_left_carry": stats(zmin["left"]), "lowest_tcp_z_right_carry": stats(zmin["right"])},
        "cube_B_init_xy": {"x": stats(rec["L_close"][:, 0]), "y": stats(rec["L_close"][:, 1]), "yaw_rad_mod_pi": stats(yaw["L_close"])},
        "cube_A_xy": {"x": stats(rec["L_open"][:, 0]), "y": stats(rec["L_open"][:, 1]), "yaw_rad_mod_pi": stats(yaw["L_open"])},
        "cube_C_init_xy": {"x": stats(rec["R_close"][:, 0]), "y": stats(rec["R_close"][:, 1]), "yaw_rad_mod_pi": stats(yaw["R_close"])},
        "right_release_xy": {"x": stats(rec["R_open"][:, 0]), "y": stats(rec["R_open"][:, 1])},
        "right_release_minus_cube_A_xy_m": stats(a_minus_ropen),
        "gripper_pitch_from_vertical_deg": {k: stats(v) for k, v in pitch.items()},
        "per_episode": {k: rec[k].round(5).tolist() for k in rec},
        "per_episode_yaw": {k: np.round(yaw[k], 4).tolist() for k in yaw},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    summary = {k: v for k, v in out.items() if not k.startswith("per_episode")}
    print(json.dumps(summary, indent=2))
    print(f"[calib] written {a.out}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
