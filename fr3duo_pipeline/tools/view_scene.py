"""Open the MuJoCo viewer on the rebuilt cube-stacking scene and watch the scripted expert.

    cd ~/ur5e_hande_workspace/hackathon
    /home/robolab-2/ur5e_hande_workspace/robot-control-stack/.venv/bin/python fr3duo_pipeline/tools/view_scene.py [--episodes 3] [--seed 20000] [--replay 0]

Needs a display (DISPLAY is set on robolab-2). The RCS GUI is a separate process that mirrors the
simulation state; the three policy cameras (head, left/right wrist) are rendered with EGL offscreen
exactly as during data generation. --replay N plays rand500 episode N's recorded commands instead
of the expert (cubes placed where the calibration says they were).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--replay", type=int, default=None, help="rand500 episode index to replay instead of the expert")
    ap.add_argument("--realtime", type=float, default=1.0, help="playback speed factor (1.0 = 6 Hz wall clock)")
    a = ap.parse_args()
    from fr3duo_pipeline.fr3duo.expert import ScriptedStacker
    from fr3duo_pipeline.fr3duo.scene import CubeScene, SceneParams

    scene = CubeScene(SceneParams(), cam_res=224, headless=False)
    scene.h.sim.open_gui()
    print("[view] GUI opened; close the window or Ctrl-C to stop")
    time.sleep(2.0)
    dt = (1.0 / 6.0) / max(a.realtime, 1e-3)
    try:
        if a.replay is not None:
            z = np.load("results/rand500_frames.npz")
            calib = json.loads(Path("fr3duo_pipeline/configs/scene_calib.json").read_text())
            P = {k: np.array(v) for k, v in calib["per_episode"].items()}
            ep = a.replay
            xy = {"cube_a": P["L_open"][ep, :2], "cube_b": P["L_close"][ep, :2], "cube_c": P["R_close"][ep, :2]}
            scene.reset(seed=1000 + ep, xy=xy)
            s0, n = z["ep_start"][ep], z["ep_len"][ep]
            for k in range(n):
                t = time.time()
                scene.step(z["action"][s0 + k])
                scene.h.sim.sync_gui()
                time.sleep(max(0.0, dt - (time.time() - t)))
            print(f"[view] replay of rand500 episode {ep}: {scene.tower_status()}")
            time.sleep(3)
            return 0
        for i in range(a.episodes):
            seed = a.seed + i
            scene.reset(seed=seed)
            ex = ScriptedStacker(scene, np.random.default_rng(seed))
            ex.reset()
            f = 0
            while not ex.done(f):
                t = time.time()
                scene.step(ex.act(f))
                scene.h.sim.sync_gui()
                f += 1
                time.sleep(max(0.0, dt - (time.time() - t)))
            for _ in range(6):
                scene.step(scene.state16())
                scene.h.sim.sync_gui()
            ts = scene.tower_status()
            print(f"[view] seed {seed}: success={ts['success']} order={ts['order']}")
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
