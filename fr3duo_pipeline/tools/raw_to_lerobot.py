"""Convert gen_native.py episodes (npz) into a LeRobot v3.0 dataset with lerobot 0.4.4.

    <envs/pi0>/bin/python fr3duo_pipeline/tools/raw_to_lerobot.py data/native_raw data/fr3_duo_native200 \
        --stats-from data/fr3_duo_rand500 --n 200

Feature schema, fps (6), task string, video codec (AV1 / libsvtav1, g=2, crf=30, yuv420p) and
camera keys match fr3_duo_rand500 so the same training command works on both. Video files roll
over at --video-mb (50 MB) instead of lerobot's 200 MB so every file stays under GitHub's 100 MB
limit; the loader does not care how episodes are spread over files.

--stats-from copies the `observation.state` and `action` entries of the source dataset's
meta/stats.json (mean/std/min/max/q01..q99) into the new dataset. Reason: lerobot's train script
re-derives the normaliser from the dataset it is given, so stage B (fine-tune on native data)
would otherwise normalise with different statistics than stage A. The native data's own stats are
kept in meta/stats_native.json for reference. Image stats stay native.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = "stack the three cubes into one tower"
CAMS = ("external_camera", "left_wrist_camera", "right_wrist_camera")


def features() -> dict:
    f = {}
    for c in CAMS:
        f[f"observation.images.{c}"] = {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}
    f["observation.state"] = {"dtype": "float32", "shape": (16,), "names": None}
    f["action"] = {"dtype": "float32", "shape": (16,), "names": None}
    f["success"] = {"dtype": "bool", "shape": (1,), "names": None}
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=0, help="use the first n episodes (sorted by seed); 0 = all")
    ap.add_argument("--stats-from", type=Path, default=None)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="async PNG writer threads (encoding is per episode)")
    ap.add_argument("--video-mb", type=int, default=50, help="roll to a new mp4 per camera above this size (GitHub rejects files > 100 MB; lerobot's default is 200)")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = sorted(a.raw.glob("ep_*.npz"))
    if a.n:
        files = files[: a.n]
    if not files:
        raise SystemExit(f"no ep_*.npz under {a.raw}")
    if a.out.exists():
        if not a.overwrite:
            raise SystemExit(f"{a.out} exists (use --overwrite)")
        shutil.rmtree(a.out)
    repo_id = a.repo_id or f"local/{a.out.name}"
    ds = LeRobotDataset.create(repo_id=repo_id, fps=6, features=features(), root=a.out, robot_type="fr3_duo", use_videos=True,
                               image_writer_threads=a.threads)
    ds.meta.update_chunk_settings(video_files_size_in_mb=a.video_mb)
    t0 = time.time()
    provenance = []
    for i, f in enumerate(files):
        z = np.load(f)
        # Read every array once: indexing the compressed NpzFile per frame re-inflates the whole
        # camera array each time (measured 0.22 s/frame, 1.5 min/episode instead of ~3 s).
        cams = {c: z[c] for c in CAMS}
        state, action, success = z["state"], z["action"], z["success"]
        n = len(state)
        for k in range(n):
            frame = {f"observation.images.{c}": cams[c][k] for c in CAMS}
            frame["observation.state"] = state[k].astype(np.float32)
            frame["action"] = action[k].astype(np.float32)
            frame["success"] = np.array([bool(success[k])])
            frame["task"] = TASK
            ds.add_frame(frame)
        ds.save_episode()
        meta = json.loads(str(z["meta"]))
        provenance.append({"episode_index": i, "source": f.name, "seed": meta["seed"], "frames": n, "spawn": meta["spawn"], "look": meta["look"]})
        print(f"[convert] {i + 1}/{len(files)} {f.name} {n} frames  ({(time.time() - t0) / 60:.1f} min)")
    ds.finalize()

    meta_dir = a.out / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    info["splits"] = {"train": f"0:{len(files)}"}
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    if a.stats_from:
        src = json.loads((a.stats_from / "meta" / "stats.json").read_text())
        dst_path = meta_dir / "stats.json"
        native = json.loads(dst_path.read_text())
        (meta_dir / "stats_native.json").write_text(json.dumps(native, indent=1))
        for key in ("observation.state", "action"):
            native[key] = src[key]
        dst_path.write_text(json.dumps(native, indent=1))
        print(f"[convert] state/action stats copied from {a.stats_from.name}; native stats kept in meta/stats_native.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    (meta_dir / "fr3duo_native_provenance.json").write_text(json.dumps({
        "generator": "fr3duo_pipeline/tools/gen_native.py + fr3duo/expert.py (scripted IK stacker, RCS FR3 Duo MuJoCo scene)",
        "lerobot_version": __import__("lerobot").__version__, "git_commit": commit, "task": TASK, "fps": 6,
        "stats_state_action_from": str(a.stats_from) if a.stats_from else "native",
        "episodes": provenance}, indent=1))
    print(f"[convert] done: {len(files)} episodes -> {a.out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
