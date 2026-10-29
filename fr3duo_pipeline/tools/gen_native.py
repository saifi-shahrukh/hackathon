"""Generate native MuJoCo cube-stacking episodes with the scripted expert (RCS venv).

    MUJOCO_GL=egl <rcs venv>/bin/python fr3duo_pipeline/tools/gen_native.py \
        --n 200 --seed0 10000 --out data/native_raw --workers 8

Each successful episode is written as <out>/ep_XXXXX.npz with
  state (T,16) float32, action (T,16) float32, success (T,) bool, timestamp (T,) float32 (k/6),
  external_camera / left_wrist_camera / right_wrist_camera (T,224,224,3) uint8,
  meta (json string: seed, spawn, look, expert schedule, tower status, attempt index).
Episodes where the expert fails are NOT written (the organizer data contains only successes) but
are counted in <out>/generation_log.jsonl so the expert's success rate is reported honestly.
Frame k holds the observation BEFORE action k is applied, as in the organizer data
(action ~ next target; |action - state| is a few mrad while moving).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CAMS = ("external_camera", "left_wrist_camera", "right_wrist_camera")


def run_one(args) -> dict:
    seed, out_dir, cam_res, scene_name = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from fr3duo_pipeline.fr3duo.expert import ScriptedStacker
    from fr3duo_pipeline.fr3duo.scene import CubeScene, SceneParams
    from fr3duo_pipeline.fr3duo.clinic_scene import ClinicScene, ClinicParams

    global _SCENE
    try:
        sc = _SCENE
    except NameError:
        sc = _SCENE = ClinicScene(ClinicParams(), cam_res=cam_res) if scene_name == "clinic" else CubeScene(SceneParams(), cam_res=cam_res)
    t0 = time.time()
    obs, info = sc.reset(seed=seed)
    ex = ScriptedStacker(sc, np.random.default_rng(seed))
    ex.reset()
    states, actions, imgs = [], [], {c: [] for c in CAMS}
    f = 0
    while not ex.done(f):
        a = ex.act(f)
        states.append(sc.state16())
        im = sc.images(obs)
        for c in CAMS:
            imgs[c].append(im[c])
        actions.append(a)
        obs = sc.step(a)
        f += 1
    for _ in range(6):
        obs = sc.step(sc.state16())
    ts = sc.tower_status()
    rec = {"seed": int(seed), "scene": scene_name, "success": bool(ts["success"]), "frames": f, "seconds": round(time.time() - t0, 1),
           "spawn": info["spawn"], "look": info["look"], "tower": {k: v for k, v in ts.items() if k != "pos"},
           "ik_err_max_mm": float(max(ex.ik_err_log) * 1000) if ex.ik_err_log else None}
    if ts["success"]:
        st = np.stack(states).astype(np.float32)
        ac = np.stack(actions).astype(np.float32)
        path = Path(out_dir) / f"ep_{seed:05d}.npz"
        np.savez_compressed(path, state=st, action=ac, success=np.ones(len(st), bool),
                            timestamp=(np.arange(len(st)) / 6.0).astype(np.float32),
                            **{c: np.stack(imgs[c]) for c in CAMS}, meta=json.dumps(rec))
        rec["file"] = str(path)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of SUCCESSFUL episodes wanted")
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=Path("data/native_raw"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cam-res", type=int, default=224)
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--scene", choices=["clinic", "flat"], default="clinic", help="clinic = organizers' room (default); flat = rand500-like floor scene")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out / "generation_log.jsonl"
    done = {json.loads(l)["seed"] for l in log.read_text().splitlines()} if log.exists() else set()
    n_ok = sum(1 for p in a.out.glob("ep_*.npz"))
    seed = a.seed0
    max_attempts = a.max_attempts or int(a.n * 1.5) + 20
    attempts = 0
    t0 = time.time()
    with Pool(a.workers) as pool, log.open("a") as lf:
        while n_ok < a.n and attempts < max_attempts:
            batch = []
            while len(batch) < a.workers and seed < a.seed0 + max_attempts:
                if seed not in done:
                    batch.append((seed, str(a.out), a.cam_res, a.scene))
                seed += 1
            if not batch:
                break
            for rec in pool.imap_unordered(run_one, batch):
                attempts += 1
                n_ok += int(rec["success"])
                lf.write(json.dumps(rec) + "\n")
                lf.flush()
                print(f"[gen] seed {rec['seed']} success={rec['success']} frames={rec['frames']} {rec['seconds']}s  "
                      f"| {n_ok}/{a.n} ok, {attempts} attempts, {(time.time() - t0) / 60:.1f} min")
    print(f"[gen] done: {n_ok} successful episodes in {a.out} after {attempts} attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
