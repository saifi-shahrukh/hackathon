"""Replay fidelity: run a rand500 episode's recorded actions open-loop in our MuJoCo scene.

    MUJOCO_GL=egl <rcs venv>/bin/python fr3duo_pipeline/tools/replay_episode.py \
        results/01_dataset_audit/rand500_frames.npz fr3duo_pipeline/configs/scene_calib.json --episodes 0-19 \
        --out results/replay

For each episode the three cubes are placed where the forward-kinematics calibration says they
were (A = left release xy, B = left grasp xy, C = right grasp xy), then the dataset's absolute
joint + gripper commands are sent at 6 Hz. If the scene matches the generator, the recorded
commands rebuild the tower. The per-episode result (tower status, final cube poses, joint
tracking error) goes to <out>/replay_<ep>.json, head/wrist frames at the key frames to
<out>/frames/, and a summary to <out>/summary.json. This number is reported in RESULTS.md as
"replay fidelity"; it bounds how much of a policy's failure can be blamed on the scene.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fr3duo_pipeline.fr3duo.scene import CubeScene, SceneParams  # noqa: E402

KEY_FRAMES = (0, 43, 109, 215, 288)


def parse_episodes(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("calib", type=Path)
    ap.add_argument("--episodes", default="0-19")
    ap.add_argument("--out", type=Path, default=Path("results/replay"))
    ap.add_argument("--cube-half", type=float, default=None)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--save-video", action="store_true", help="save the head camera of every frame as PNGs")
    a = ap.parse_args()

    z = np.load(a.npz)
    st, ac, ev, ep_start, ep_len = z["state"], z["action"], z["events"], z["ep_start"], z["ep_len"]
    calib = json.loads(a.calib.read_text())
    P = {k: np.array(v) for k, v in calib["per_episode"].items()}
    params = SceneParams()
    if a.cube_half:
        params.cube_half = a.cube_half
    scene = CubeScene(params, cam_res=224)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frames").mkdir(exist_ok=True)
    from PIL import Image

    results = []
    for ep in parse_episodes(a.episodes):
        t0 = time.time()
        xy = {"cube_a": P["L_open"][ep, :2], "cube_b": P["L_close"][ep, :2], "cube_c": P["R_close"][ep, :2]}
        obs, info = scene.reset(seed=1000 + ep, xy=xy)
        s0, n = ep_start[ep], ep_len[ep]
        track_err = []
        status_at = {}
        for k in range(n):
            if not a.no_frames and (k in KEY_FRAMES or k == n - 1 or a.save_video):
                im = scene.images(obs)
                tag = f"ep{ep:03d}_f{k:03d}"
                if a.save_video or k in KEY_FRAMES or k == n - 1:
                    Image.fromarray(np.concatenate([im["external_camera"], im["left_wrist_camera"], im["right_wrist_camera"]], 1)).save(a.out / "frames" / f"{tag}.png")
            obs = scene.step(ac[s0 + k])
            s_sim = scene.state16()
            track_err.append(np.abs(s_sim[[*range(0, 7), *range(8, 15)]] - st[s0 + k][[*range(0, 7), *range(8, 15)]]).max())
            if k in (ev[ep, 1] + 10, ev[ep, 3] + 10):   # shortly after each release
                status_at[int(k)] = scene.tower_status()
        # settle and judge
        for _ in range(6):
            obs = scene.step(scene.state16())
        final = scene.tower_status()
        if not a.no_frames:
            im = scene.images(obs)
            Image.fromarray(np.concatenate([im["external_camera"], im["left_wrist_camera"], im["right_wrist_camera"]], 1)).save(a.out / "frames" / f"ep{ep:03d}_final.png")
        rec = {"episode": int(ep), "success": final["success"], "final": final, "status_after_releases": status_at,
               "spawn": info["spawn"], "track_err_rad": {"median": float(np.median(track_err)), "p95": float(np.percentile(track_err, 95)), "max": float(np.max(track_err))},
               "frames": int(n), "seconds": round(time.time() - t0, 1)}
        (a.out / f"replay_{ep:03d}.json").write_text(json.dumps(rec, indent=1))
        results.append(rec)
        print(f"[replay] ep {ep:3d}: success={final['success']}  order={final['order']}  on_table={final['on_table']} mid={final['mid_ok']} top={final['top_ok']} "
              f"still={final['still']} released={final['released']}  track p95 {rec['track_err_rad']['p95']:.4f} rad  {rec['seconds']} s")
    n_ok = sum(r["success"] for r in results)
    summ = {"episodes": [r["episode"] for r in results], "n": len(results), "success": n_ok, "rate": n_ok / max(1, len(results)),
            "cube_half": params.cube_half, "per_episode": [{k: r[k] for k in ("episode", "success", "track_err_rad")} for r in results]}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[replay] {n_ok}/{len(results)} episodes rebuilt the tower (cube_half={params.cube_half})")
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
