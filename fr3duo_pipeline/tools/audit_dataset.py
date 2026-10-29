"""Audit a LeRobot v3.0 FR3 Duo dataset from its parquet files and write results/<name>_audit.json.

    <lerobot venv>/bin/python fr3duo_pipeline/tools/audit_dataset.py data/fr3_duo_rand500

Reads only pandas/pyarrow (no lerobot import, so it also runs in the RCS venv). Everything the
scene calibration and the scripted expert rely on is measured here, not assumed:
fps and timestamp spacing, episode lengths, the home pose and its distance to robot-control-stack's
FR3 Duo home, gripper open/close event frames per arm, action-vs-state offsets, peak joint speeds,
state ranges, gripper hold values, stats keys (q01/q99 for pi0.5), the task string, and the
`splits` field (known wrong in rand500).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# robot-control-stack python/rcs/__init__.py HOME_POSITIONS (verified 2026-10-29)
RCS_HOME_LEFT = np.array([0.48797692, -0.57224476, -0.58536988, -2.57958827, 0.86400183, 2.0530809, -0.85965005])
RCS_HOME_RIGHT = np.array([-0.48797676, -0.57224472, 0.58536959, -2.57958788, -0.86400148, 2.05308196, 0.85965057])
L_Q, L_G, R_Q, R_G = slice(0, 7), 7, slice(8, 15), 15


def gripper_events(cmd: np.ndarray, thr: float = 0.5) -> list[tuple[int, str]]:
    """Frames where the binary gripper command flips (1 = open, 0 = close)."""
    closed = (cmd < thr).astype(int)
    out = []
    for i in np.flatnonzero(np.diff(closed)):
        out.append((int(i + 1), "close" if closed[i + 1] else "open"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = a.root
    info = json.loads((root / "meta" / "info.json").read_text())
    stats = json.loads((root / "meta" / "stats.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    st = np.stack(df["observation.state"].to_numpy())
    ac = np.stack(df["action"].to_numpy())
    ep = df["episode_index"].to_numpy()
    ts = df["timestamp"].to_numpy()
    fps = info["fps"]

    lens = df.groupby("episode_index").size()
    dts = []
    for e, g in df.groupby("episode_index"):
        dts.append(np.diff(g["timestamp"].to_numpy()))
    dts = np.concatenate(dts)

    first = df.groupby("episode_index").head(1)
    s0 = np.stack(first["observation.state"].to_numpy())
    home_err_l = np.abs(s0[:, L_Q] - RCS_HOME_LEFT).max()
    home_err_r = np.abs(s0[:, R_Q] - RCS_HOME_RIGHT).max()

    ev_l, ev_r, speeds = [], [], []
    for e, g in df.groupby("episode_index"):
        a_e = np.stack(g["action"].to_numpy())
        s_e = np.stack(g["observation.state"].to_numpy())
        ev_l.append(gripper_events(a_e[:, L_G]))
        ev_r.append(gripper_events(a_e[:, R_G]))
        v = np.abs(np.diff(s_e[:, list(range(0, 7)) + list(range(8, 15))], axis=0)) * fps
        speeds.append(v.max(0))
    speeds = np.stack(speeds)

    def event_table(evs):
        close = [e[0][0] for e in evs if len(e) >= 1 and e[0][1] == "close"]
        open_ = [e[1][0] for e in evs if len(e) >= 2 and e[1][1] == "open"]
        return {
            "episodes_with_exactly_close_then_open": int(sum(len(e) == 2 and e[0][1] == "close" for e in evs)),
            "n_events_histogram": {str(k): int(v) for k, v in zip(*np.unique([len(e) for e in evs], return_counts=True))},
            "close_frame": {"median": float(np.median(close)), "min": int(min(close)), "max": int(max(close))} if close else None,
            "open_frame": {"median": float(np.median(open_)), "min": int(min(open_)), "max": int(max(open_))} if open_ else None,
        }

    grip_state_l, grip_state_r = st[:, L_G], st[:, R_G]
    hold_l = grip_state_l[ac[:, L_G] < 0.5]
    hold_r = grip_state_r[ac[:, R_G] < 0.5]
    ad = np.abs(ac - st)

    report = {
        "dataset": root.name,
        "codebase_version": info["codebase_version"],
        "fps": fps,
        "timestamp_dt": {"mean": float(dts.mean()), "min": float(dts.min()), "max": float(dts.max()), "expected": 1.0 / fps},
        "total_episodes": int(ep.max() + 1), "total_frames": int(len(df)),
        "info_total_episodes": info["total_episodes"], "info_total_frames": info["total_frames"],
        "info_splits": info.get("splits"), "splits_consistent": info.get("splits", {}).get("train") == f"0:{int(ep.max() + 1)}",
        "tasks": tasks.index.tolist() if "task" not in tasks.columns else tasks["task"].tolist(),
        "success_all_true": bool(df["success"].all()) if "success" in df else None,
        "episode_length": {"mean": float(lens.mean()), "std": float(lens.std()), "min": int(lens.min()), "max": int(lens.max()),
                           "seconds_mean": float(lens.mean() / fps)},
        "state_dim": int(st.shape[1]), "action_dim": int(ac.shape[1]),
        "home_pose_frame0": {"left": s0[0, L_Q].round(5).tolist(), "right": s0[0, R_Q].round(5).tolist(),
                             "max_abs_err_vs_rcs_home_left": float(home_err_l), "max_abs_err_vs_rcs_home_right": float(home_err_r),
                             "frame0_spread_across_episodes": float(np.abs(s0 - s0[0]).max())},
        "gripper": {
            "action_unique_values": sorted(set(np.unique(ac[:, [L_G, R_G]].round(3)).tolist())),
            "state_range_left": [float(grip_state_l.min()), float(grip_state_l.max())],
            "state_range_right": [float(grip_state_r.min()), float(grip_state_r.max())],
            "state_while_commanded_closed": {"left_median": float(np.median(hold_l)) if len(hold_l) else None,
                                             "right_median": float(np.median(hold_r)) if len(hold_r) else None,
                                             "left_p05_p95": [float(np.percentile(hold_l, 5)), float(np.percentile(hold_l, 95))] if len(hold_l) else None},
            "events_left": event_table(ev_l), "events_right": event_table(ev_r),
        },
        "action_minus_state_abs": {"mean_per_dim": ad.mean(0).round(4).tolist(), "p99_joints": float(np.percentile(ad[:, list(range(0, 7)) + list(range(8, 15))], 99))},
        "peak_joint_speed_rad_s": {"per_joint_max": speeds.max(0).round(3).tolist(), "per_episode_max_median": float(np.median(speeds.max(1)))},
        "state_min": st.min(0).round(4).tolist(), "state_max": st.max(0).round(4).tolist(),
        "stats_keys": {k: sorted(stats[k].keys()) for k in ("observation.state", "action") if k in stats},
        "has_quantiles": all({"q01", "q99"} <= set(stats.get(k, {})) for k in ("observation.state", "action")),
        "image_features": {k: v["shape"] for k, v in info["features"].items() if v["dtype"] == "video"},
        "video_codec": next((v["info"].get("video.codec") for v in info["features"].values() if v["dtype"] == "video"), None),
    }
    out = a.out or Path("results") / f"{root.name}_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
