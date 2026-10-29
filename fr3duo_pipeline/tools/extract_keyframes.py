"""Dump a LeRobot v3.0 FR3 Duo dataset's states/actions and per-episode gripper events to one npz.

    python fr3duo_pipeline/tools/extract_keyframes.py data/fr3_duo_rand500 results/01_dataset_audit/rand500_frames.npz

The RCS venv (MuJoCo side) has no pandas/pyarrow, so the calibration and replay tools read this
npz instead of the parquet files. Contents: state (N,16) float32, action (N,16) float32,
episode_index (N,), frame_index (N,), ep_start (E,), ep_len (E,), and for each episode the four
gripper event frames [left_close, left_open, right_close, right_open] (E,4) from the binary gripper
command (1 = open, 0 = closed). Frame k in the npz is the k-th row of the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    ac = np.stack(df["action"].to_numpy()).astype(np.float32)
    ep = df["episode_index"].to_numpy().astype(np.int64)
    fr = df["frame_index"].to_numpy().astype(np.int64)
    eps = np.unique(ep)
    ep_start = np.array([np.flatnonzero(ep == e)[0] for e in eps])
    ep_len = np.array([(ep == e).sum() for e in eps])
    events = np.full((len(eps), 4), -1, dtype=np.int64)
    for i, e in enumerate(eps):
        a = ac[ep_start[i]: ep_start[i] + ep_len[i]]
        for j, col in enumerate((7, 15)):
            closed = (a[:, col] < 0.5).astype(int)
            flips = np.flatnonzero(np.diff(closed)) + 1
            if len(flips) >= 1:
                events[i, 2 * j] = flips[0]
            if len(flips) >= 2:
                events[i, 2 * j + 1] = flips[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=st, action=ac, episode_index=ep, frame_index=fr,
                        ep_start=ep_start, ep_len=ep_len, events=events, fps=np.int64(6))
    print(f"{out}: {len(eps)} episodes, {len(st)} frames, events ok in "
          f"{int((events >= 0).all(1).sum())} episodes; event medians {np.median(events, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
