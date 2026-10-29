"""Tile organizer-dataset frames next to our replay renders at the key frames (lerobot 0.4.4 venv).

    <envs/pi0>/bin/python fr3duo_pipeline/tools/side_by_side.py data/fr3_duo_rand500 results/02_scene_replay/episodes_00-09 --episodes 0,1,2

Output <replay_dir>/side_by_side_epXXX.png: one row per key frame (0, 43, 109, 215, 288, final),
left = organizer dataset (head | left wrist | right wrist), right = our MuJoCo replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEY = (0, 43, 109, 215, 288)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("replay_dir", type=Path)
    ap.add_argument("--episodes", default="0,1,2")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    ds = LeRobotDataset(f"local/{a.dataset.name}", root=a.dataset, video_backend="pyav")
    cams = ("external_camera", "left_wrist_camera", "right_wrist_camera")
    for ep in [int(e) for e in a.episodes.split(",")]:
        start = int(ds.meta.episodes["dataset_from_index"][ep])
        length = int(ds.meta.episodes["length"][ep])
        rows = []
        for k in list(KEY) + [length - 1]:
            item = ds[start + k]
            left = np.concatenate([(item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in cams], 1)
            name = f"ep{ep:03d}_f{k:03d}.png" if k in KEY else f"ep{ep:03d}_final.png"
            p = a.replay_dir / "frames" / name
            right = np.array(Image.open(p).convert("RGB")) if p.exists() else np.zeros_like(left)
            rows.append(np.concatenate([left, np.full((224, 8, 3), 255, np.uint8), right], 1))
        out = a.replay_dir / f"side_by_side_ep{ep:03d}.png"
        Image.fromarray(np.concatenate(rows, 0)).save(out)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
