"""Closed-loop evaluation of a served policy (or the scripted expert) in the CubeScene (RCS venv).

    # policy: start fr3duo_pipeline/eval/policy_server.py in the lerobot venv first, then
    MUJOCO_GL=egl <rcs venv>/bin/python fr3duo_pipeline/eval/run_eval.py --condition A_5k \
        --n-action-steps 10 --out results/eval
    # expert sanity condition (no server):
    MUJOCO_GL=egl <rcs venv>/bin/python fr3duo_pipeline/eval/run_eval.py --condition expert --expert --out results/eval

Protocol: every condition uses the same seeds (configs/eval_seeds.json), a 360-frame cap at 6 Hz,
chunked execution (the policy is queried whenever its queue is empty; the first n_action_steps
actions of each predicted chunk are executed one per frame), success = CubeScene.tower_status()
at the end (3-cube tower, settled, released). Output: <out>/<condition>/episodes.jsonl, summary.json,
and head-camera frame strips for the first 5 seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fr3duo_pipeline.eval.bridge import AUTHKEY, PORT, pack_obs, unpack  # noqa: E402

TASK = "stack the three cubes into one tower"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--seeds", type=Path, default=Path("fr3duo_pipeline/configs/eval_seeds.json"))
    ap.add_argument("--n-seeds", type=int, default=0, help="0 = all seeds in the file")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--out", type=Path, default=Path("results/eval"))
    ap.add_argument("--strips", type=int, default=5, help="save head-camera strips for this many seeds")
    ap.add_argument("--scene", choices=["clinic", "flat"], default="clinic")
    a = ap.parse_args()
    from PIL import Image
    from fr3duo_pipeline.fr3duo.scene import CubeScene, SceneParams
    from fr3duo_pipeline.fr3duo.clinic_scene import ClinicScene, ClinicParams

    seeds = json.loads(a.seeds.read_text())["seeds"]
    if a.n_seeds:
        seeds = seeds[: a.n_seeds]
    out = a.out / a.condition
    out.mkdir(parents=True, exist_ok=True)
    (out / "strips").mkdir(exist_ok=True)
    scene = ClinicScene(ClinicParams(), cam_res=224) if a.scene == "clinic" else CubeScene(SceneParams(), cam_res=224)
    conn = None
    info = {}
    if not a.expert:
        conn = Client(("127.0.0.1", a.port), authkey=AUTHKEY)
        conn.send({"cmd": "info"})
        info = conn.recv()
        print(f"[eval] server: {info}")
        if a.n_action_steps > info["chunk_size"]:
            raise SystemExit(f"n_action_steps {a.n_action_steps} > chunk_size {info['chunk_size']}")
    jl = (out / "episodes.jsonl").open("w")
    n_ok = 0
    t_all = time.time()
    for i, seed in enumerate(seeds):
        obs, sinfo = scene.reset(seed=seed)
        if a.expert:
            from fr3duo_pipeline.fr3duo.expert import ScriptedStacker
            ex = ScriptedStacker(scene, np.random.default_rng(seed))
            ex.reset()
        else:
            conn.send({"cmd": "reset"})
            conn.recv()
        queue = []
        strip = []
        infer_ms = []
        grip_events = {"left": [], "right": []}
        prev_g = [1.0, 1.0]
        t0 = time.time()
        f = 0
        first_success_frame = None
        while f < a.max_frames:
            if a.expert:
                if ex.done(f):
                    break
                action = ex.act(f)
            else:
                if not queue:
                    conn.send({"cmd": "act", "obs": pack_obs(scene.images(obs), scene.state16()), "task": TASK})
                    rep = conn.recv()
                    chunk = unpack(rep["chunk"])
                    infer_ms.append(rep["ms"])
                    queue = [chunk[k] for k in range(min(a.n_action_steps, len(chunk)))]
                action = np.asarray(queue.pop(0), dtype=np.float32)
                action[7] = 1.0 if action[7] >= 0.5 else 0.0
                action[15] = 1.0 if action[15] >= 0.5 else 0.0
            for j, gi in enumerate((7, 15)):
                if (action[gi] < 0.5) != (prev_g[j] < 0.5):
                    grip_events["left" if j == 0 else "right"].append(f)
                prev_g[j] = float(action[gi])
            obs = scene.step(action)
            if i < a.strips and f % 30 == 0:
                strip.append(scene.images(obs)["external_camera"])
            if f % 12 == 0 and first_success_frame is None and scene.tower_status()["success"]:
                first_success_frame = f
            f += 1
        for _ in range(6):
            obs = scene.step(scene.state16())
        ts = scene.tower_status()
        n_ok += int(ts["success"])
        rec = {"seed": int(seed), "success": bool(ts["success"]), "frames": f, "first_success_frame": first_success_frame,
               "tower": {k: v for k, v in ts.items() if k != "pos"}, "final_pos": ts["pos"], "spawn": sinfo["spawn"],
               "grip_events": grip_events, "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
               "n_action_steps": a.n_action_steps, "seconds": round(time.time() - t0, 1)}
        jl.write(json.dumps(rec) + "\n")
        jl.flush()
        if strip:
            strip.append(scene.images(obs)["external_camera"])
            Image.fromarray(np.concatenate(strip[:13], 1)).save(out / "strips" / f"seed{seed}_{'ok' if ts['success'] else 'fail'}.png")
        print(f"[eval] {a.condition} seed {seed} ({i + 1}/{len(seeds)}): success={ts['success']} order={ts['order']} "
              f"grip L{grip_events['left'][:4]} R{grip_events['right'][:4]} frames={f} {rec['seconds']}s | {n_ok}/{i + 1}")
    jl.close()
    lo, hi = wilson(n_ok, len(seeds))
    summ = {"condition": a.condition, "n": len(seeds), "success": n_ok, "rate": n_ok / len(seeds), "wilson95": [lo, hi],
            "n_action_steps": a.n_action_steps, "max_frames": a.max_frames, "scene": a.scene, "server": info, "seeds": seeds,
            "minutes": round((time.time() - t_all) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[eval] {a.condition}: {n_ok}/{len(seeds)} = {100 * n_ok / len(seeds):.0f} % (Wilson 95 % {100 * lo:.0f}-{100 * hi:.0f} %)")
    if conn is not None:
        conn.close()
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
