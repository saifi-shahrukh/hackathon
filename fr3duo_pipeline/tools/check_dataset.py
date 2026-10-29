"""Parity check between the native MuJoCo dataset and the organizer dataset (lerobot 0.4.4 venv).

    <envs/pi0>/bin/python fr3duo_pipeline/tools/check_dataset.py data/fr3_duo_native200 data/fr3_duo_rand500 \
        --out results/native_vs_rand500

Writes <out>/check.json (schema equality, fps, counts, value ranges, gripper action values,
decode test) and <out>/side_by_side.png (first frames of 4 episodes, native on top, rand500 below,
head | left wrist | right wrist). Exit code 1 if a hard check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("native", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/native_vs_rand500"))
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    a.out.mkdir(parents=True, exist_ok=True)
    nat = LeRobotDataset(f"local/{a.native.name}", root=a.native, video_backend="pyav")
    ref = LeRobotDataset(f"local/{a.reference.name}", root=a.reference, video_backend="pyav")
    ni, ri = nat.meta.info, ref.meta.info
    strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk != "info"} for k, v in d.items()}  # noqa: E731
    checks = {
        "features_equal": strip(ni["features"]) == strip(ri["features"]),
        "fps": (ni["fps"], ri["fps"]), "codebase_version": (ni["codebase_version"], ri["codebase_version"]),
        "episodes": (nat.num_episodes, ref.num_episodes), "frames": (nat.num_frames, ref.num_frames),
        "tasks_equal": list(nat.meta.tasks.index) == list(ref.meta.tasks.index),
        "video_codec": (ni["features"]["observation.images.external_camera"]["info"].get("video.codec"),
                        ri["features"]["observation.images.external_camera"]["info"].get("video.codec")),
        "state_action_stats_identical": all(np.allclose(nat.meta.stats[k][q], ref.meta.stats[k][q]) for k in ("observation.state", "action") for q in ("mean", "std", "q01", "q99")),
    }
    # value ranges from the parquet data
    import pandas as pd
    def load(root):
        return pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("chunk-*/file-*.parquet"))], ignore_index=True)
    dn, dr = load(a.native), load(a.reference)
    sn, sr = np.stack(dn["observation.state"].to_numpy()), np.stack(dr["observation.state"].to_numpy())
    an = np.stack(dn["action"].to_numpy())
    checks["native_state_within_ref_range_frac"] = float(((sn >= sr.min(0) - 0.05) & (sn <= sr.max(0) + 0.05)).all(1).mean())
    checks["native_gripper_action_values"] = sorted(set(np.unique(an[:, [7, 15]]).round(3).tolist()))
    checks["native_episode_len"] = {"mean": float(dn.groupby("episode_index").size().mean()), "min": int(dn.groupby("episode_index").size().min()), "max": int(dn.groupby("episode_index").size().max())}
    checks["ref_episode_len_mean"] = float(dr.groupby("episode_index").size().mean())
    checks["native_success_all"] = bool(dn["success"].all())
    # decode test + side by side
    rows = []
    eps_n = [int(x) for x in np.linspace(0, nat.num_episodes - 1, 4)]
    eps_r = [0, 1, 2, 3]
    for en, er in zip(eps_n, eps_r):
        fn = nat[int(nat.meta.episodes["dataset_from_index"][en])]
        fr = ref[int(ref.meta.episodes["dataset_from_index"][er])]
        def tile(item):
            return np.concatenate([(item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                                   for c in ("external_camera", "left_wrist_camera", "right_wrist_camera")], 1)
        rows.append(np.concatenate([tile(fn), tile(fr)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(a.out / "side_by_side.png")
    checks["decode_ok"] = True
    # image brightness parity (head camera, 20 random frames each)
    rng = np.random.default_rng(0)
    bn = [float(nat[int(i)]["observation.images.external_camera"].mean()) for i in rng.integers(0, nat.num_frames, 20)]
    br = [float(ref[int(i)]["observation.images.external_camera"].mean()) for i in rng.integers(0, ref.num_frames, 20)]
    checks["head_mean_brightness"] = {"native": float(np.mean(bn)), "ref": float(np.mean(br))}
    hard = checks["features_equal"] and checks["fps"][0] == checks["fps"][1] and checks["tasks_equal"] and checks["decode_ok"] \
        and checks["native_gripper_action_values"] == [0.0, 1.0] and checks["native_success_all"]
    checks["PASS"] = bool(hard)
    (a.out / "check.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0 if hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
