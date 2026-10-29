"""Watch the rebuilt cube-stacking scene: the scripted expert, or a rand500 replay.

    cd ~/ur5e_hande_workspace/hackathon
    /home/robolab-2/ur5e_hande_workspace/robot-control-stack/.venv/bin/python fr3duo_pipeline/tools/view_scene.py            # cameras window
    ... view_scene.py --mode gui          # the interactive RCS/MuJoCo viewer (needs working GLX on your display)
    ... view_scene.py --replay 0          # replay rand500 episode 0's recorded commands (cubes at the calibrated spots)
    ... view_scene.py --scene flat        # the rand500-look floor scene instead of the organizers' room

--mode cameras (default) opens one OpenCV window showing exactly what the policy sees (head, left
wrist, right wrist, 224x224, rendered offscreen with EGL as during data generation) plus a free
overview camera. It needs no GLX, so it also works in VNC sessions. Press q or Esc to quit.
--mode gui starts RCS's MuJoCo viewer in a separate process (mujoco.viewer, GLFW/GLX).
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

# GL environment. robolab-2's ~/.bashrc exports __NV_PRIME_RENDER_OFFLOAD=1 and friends; depending on
# the terminal these make GLX refuse a context ("BadValue") or let mesa's EGL grab the NVIDIA device
# ("driver (null)", "failed to create dri2 screen"). The dynamic loader and glvnd read these at process
# start, so the script re-executes itself once with a cleaned environment before any GL import.
_MODE = "gui" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "gui" else "cameras"
_GL_DROP = ("__NV_PRIME_RENDER_OFFLOAD", "__VK_LAYER_NV_optimus", "LIBGL_ALWAYS_SOFTWARE", "LIBGL_ALWAYS_INDIRECT",
            "PYOPENGL_PLATFORM", "LD_PRELOAD")
if os.environ.get("FR3DUO_GL_SANITIZED") != "1":
    env = {k: v for k, v in os.environ.items() if k not in _GL_DROP}
    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    env["MUJOCO_GL"] = "glfw" if _MODE == "gui" else "egl"
    if _MODE != "gui" and os.path.exists("/usr/share/glvnd/egl_vendor.d/10_nvidia.json"):
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    env["FR3DUO_GL_SANITIZED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
STOP = {"flag": False}


def _on_sigint(_signum, _frame):  # first Ctrl-C: stop the loop and close cleanly; second: hard exit
    if STOP["flag"]:
        os._exit(130)
    STOP["flag"] = True
    print("\n[view] stopping (press Ctrl-C again to force)", flush=True)


class CameraWindow:
    """OpenCV window: policy cameras (x2) on top, a free overview camera below."""

    def __init__(self, scene, title: str, snapshot: Path | None = None):
        import cv2
        import mujoco
        self.cv2, self.mujoco = cv2, mujoco
        self.scene = scene
        self.title = title
        self.snapshot = snapshot
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        self.quit = False

    def show(self, obs, caption: str) -> None:
        cv2, np_ = self.cv2, np
        im = self.scene.images(obs)
        top = np_.concatenate([cv2.resize(im[k], (224, 224)) for k in ("external_camera", "left_wrist_camera", "right_wrist_camera")], 1)
        top = cv2.resize(top, (672, 224))
        over = cv2.resize(im["overview"], (672, 378)) if "overview" in im else np_.zeros((378, 672, 3), np_.uint8)
        frame = np_.concatenate([top, over], 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, caption, (8, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "head | left wrist | right wrist (policy input)   q/Esc: quit", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(self.title, frame)
        self.n_shown = getattr(self, "n_shown", 0) + 1
        if self.snapshot is not None and self.n_shown % 30 == 1:
            cv2.imwrite(str(self.snapshot), frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            self.quit = True

    def close(self):
        try:
            self.cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cameras", "gui"], default="cameras")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--replay", type=int, default=None, help="rand500 episode index to replay instead of the expert")
    ap.add_argument("--realtime", type=float, default=1.0, help="playback speed factor (1.0 = 6 Hz wall clock)")
    ap.add_argument("--scene", choices=["clinic", "flat"], default="clinic", help="clinic = organizers' room (default)")
    ap.add_argument("--snapshot", type=Path, default=None, help="also save the composed cameras frame to this PNG every 30 frames")
    a = ap.parse_args()
    from fr3duo_pipeline.fr3duo.clinic_scene import ClinicParams, ClinicScene
    from fr3duo_pipeline.fr3duo.expert import ScriptedStacker
    from fr3duo_pipeline.fr3duo.scene import CubeScene, SceneParams

    signal.signal(signal.SIGINT, _on_sigint)
    gui = a.mode == "gui"
    t_build = time.time()
    print(f"[view] building the {a.scene} scene (5 s idle, up to a minute while data generation or training use the GPU)...", flush=True)
    scene = (ClinicScene(ClinicParams(), cam_res=224, headless=not gui, overview=not gui) if a.scene == "clinic"
             else CubeScene(SceneParams(), cam_res=224, headless=not gui, overview=not gui))   # headless=False opens the RCS GUI process
    print(f"[view] scene ready in {time.time() - t_build:.1f} s", flush=True)
    win = None
    if gui:
        print("[view] RCS GUI process started (a MuJoCo window should appear); Ctrl-C to stop", flush=True)
        time.sleep(2.0)
    else:
        win = CameraWindow(scene, f"fr3duo {a.scene} scene", a.snapshot)
        print("[view] cameras window opened; q/Esc or Ctrl-C to stop", flush=True)
    dt = (1.0 / 6.0) / max(a.realtime, 1e-3)

    def stop() -> bool:
        return STOP["flag"] or (win is not None and win.quit)

    def tick(obs, caption):
        if win is not None:
            win.show(obs, caption)

    def pace(t):
        time.sleep(max(0.0, dt - (time.time() - t)))

    try:
        if a.replay is not None:
            z = np.load("results/01_dataset_audit/rand500_frames.npz")
            calib = json.loads(Path("fr3duo_pipeline/configs/scene_calib.json").read_text())
            P = {k: np.array(v) for k, v in calib["per_episode"].items()}
            ep = a.replay
            xy = {"cube_a": P["L_open"][ep, :2], "cube_b": P["L_close"][ep, :2], "cube_c": P["R_close"][ep, :2]}
            if a.scene == "clinic":
                print("[view] note: rand500 replay assumes the flat scene geometry; use --scene flat for a faithful replay")
            obs, _ = scene.reset(seed=1000 + ep, xy=xy)
            s0, n = z["ep_start"][ep], z["ep_len"][ep]
            for k in range(n):
                if stop():
                    break
                t = time.time()
                obs = scene.step(z["action"][s0 + k])
                tick(obs, f"rand500 replay ep {ep}  frame {k}/{n}")
                pace(t)
            print(f"[view] replay of rand500 episode {ep}: success={scene.tower_status()['success']}", flush=True)
            time.sleep(2)
        else:
            for i in range(a.episodes):
                if stop():
                    break
                seed = a.seed + i
                obs, _ = scene.reset(seed=seed)
                if os.environ.get("FR3DUO_VIEW_DEBUG"):
                    print("[view] reset done", flush=True)
                ex = ScriptedStacker(scene, np.random.default_rng(seed))
                ex.reset()
                f = 0
                while not ex.done(f) and not stop():
                    t = time.time()
                    obs = scene.step(ex.act(f))
                    tick(obs, f"expert  seed {seed}  frame {f}/{ex.total_frames}")
                    if f < 3 and os.environ.get("FR3DUO_VIEW_DEBUG"):
                        print(f"[view] frame {f} shown ({time.time() - t:.2f} s)", flush=True)
                    f += 1
                    pace(t)
                for _ in range(6):
                    obs = scene.step(scene.state16())
                ts = scene.tower_status()
                tick(obs, f"seed {seed}: success={ts['success']}")
                print(f"[view] seed {seed}: success={ts['success']} order={ts['order']}", flush=True)
                time.sleep(2)
    finally:
        print("[view] closing", flush=True)
        if win is not None:
            win.close()
        if gui:
            try:
                scene.h.sim.close_gui()
            except Exception as e:  # noqa: BLE001
                print(f"[view] close_gui: {type(e).__name__}: {e}")
        os._exit(0)   # the RCS env's own close() can hang on the GUI/IPC teardown; exit directly


if __name__ == "__main__":
    raise SystemExit(main())
