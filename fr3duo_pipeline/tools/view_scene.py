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
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# The RCS GUI is mujoco.viewer in a spawned process and needs a real GL window: with MUJOCO_GL=egl
# (the default our scene sets for headless rendering) no window ever appears. Set GLFW before the
# scene module is imported; the three offscreen policy cameras render fine through GLFW too.
os.environ["MUJOCO_GL"] = "glfw"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
STOP = {"flag": False}


def _on_sigint(signum, frame):  # first Ctrl-C: stop the loop and close cleanly; second: hard exit
    if STOP["flag"]:
        os._exit(130)
    STOP["flag"] = True
    print("\n[view] stopping (press Ctrl-C again to force)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--replay", type=int, default=None, help="rand500 episode index to replay instead of the expert")
    ap.add_argument("--realtime", type=float, default=1.0, help="playback speed factor (1.0 = 6 Hz wall clock)")
    ap.add_argument("--scene", choices=["clinic", "flat"], default="clinic", help="clinic = organizers' room (default)")
    a = ap.parse_args()
    from fr3duo_pipeline.fr3duo.expert import ScriptedStacker
    from fr3duo_pipeline.fr3duo.scene import CubeScene, SceneParams
    from fr3duo_pipeline.fr3duo.clinic_scene import ClinicScene, ClinicParams

    signal.signal(signal.SIGINT, _on_sigint)
    scene = (ClinicScene(ClinicParams(), cam_res=224, headless=False) if a.scene == "clinic"
             else CubeScene(SceneParams(), cam_res=224, headless=False))   # headless=False opens the RCS GUI process
    print("[view] GUI process started (a MuJoCo window should appear); Ctrl-C to stop", flush=True)
    time.sleep(2.0)
    dt = (1.0 / 6.0) / max(a.realtime, 1e-3)

    def pace(t):
        time.sleep(max(0.0, dt - (time.time() - t)))

    try:
        if a.replay is not None:
            z = np.load("results/01_dataset_audit/rand500_frames.npz")
            calib = json.loads(Path("fr3duo_pipeline/configs/scene_calib.json").read_text())
            P = {k: np.array(v) for k, v in calib["per_episode"].items()}
            ep = a.replay
            xy = {"cube_a": P["L_open"][ep, :2], "cube_b": P["L_close"][ep, :2], "cube_c": P["R_close"][ep, :2]}
            scene.reset(seed=1000 + ep, xy=xy)
            s0, n = z["ep_start"][ep], z["ep_len"][ep]
            for k in range(n):
                if STOP["flag"]:
                    break
                t = time.time()
                scene.step(z["action"][s0 + k])
                pace(t)
            print(f"[view] replay of rand500 episode {ep}: success={scene.tower_status()['success']}", flush=True)
            time.sleep(3)
        else:
            for i in range(a.episodes):
                if STOP["flag"]:
                    break
                seed = a.seed + i
                scene.reset(seed=seed)
                ex = ScriptedStacker(scene, np.random.default_rng(seed))
                ex.reset()
                f = 0
                while not ex.done(f) and not STOP["flag"]:
                    t = time.time()
                    scene.step(ex.act(f))
                    f += 1
                    pace(t)
                for _ in range(6):
                    scene.step(scene.state16())
                ts = scene.tower_status()
                print(f"[view] seed {seed}: success={ts['success']} order={ts['order']}", flush=True)
                time.sleep(2)
    finally:
        print("[view] closing", flush=True)
        try:
            scene.h.sim.close_gui()
        except Exception as e:  # noqa: BLE001
            print(f"[view] close_gui: {type(e).__name__}: {e}")
        os._exit(0)   # the RCS env's own close() can hang on the GUI/IPC teardown; exit directly


if __name__ == "__main__":
    raise SystemExit(main())
