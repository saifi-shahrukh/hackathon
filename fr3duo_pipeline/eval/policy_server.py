"""Serve a pi0.5 checkpoint (lerobot 0.4.4 venv) to the MuJoCo evaluation loop over localhost.

    <envs/pi0>/bin/python fr3duo_pipeline/eval/policy_server.py \
        --ckpt checkpoints/A_pi05_rand500_5k/checkpoints/005000/pretrained_model [--port 6111] [--dtype bfloat16]

    # the zero-fine-tune "base" condition: the patched pi05_base with the dataset's stats and renames
    <envs/pi0>/bin/python fr3duo_pipeline/eval/policy_server.py --base $HF_HOME/patched_bases/pi05_base \
        --dataset-root data/fr3_duo_rand500

Protocol (multiprocessing.connection, see bridge.py): {"cmd": "reset"} -> "ok";
{"cmd": "act", "obs": packed obs, "task": str} -> packed (chunk_size, 16) float32 absolute actions;
{"cmd": "info"} -> dict; {"cmd": "quit"}.

LoRA checkpoints are merged onto their base on the CPU first (pi0.5 is ~14 GB in float32; the
RTX 5070 Ti has 16 GB), then cast to bfloat16 and moved to the GPU (~7 GB). The adapter's
`base_model_name_or_path` is an absolute path from the training machine; --base-override maps it.
Pattern follows the thesis `src/vla/models/adapters/lerobot_adapter.py` (lerobot 0.4.4).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fr3duo_pipeline.eval.bridge import AUTHKEY, PORT, pack, unpack_obs  # noqa: E402

RENAME = {"observation.images.external_camera": "observation.images.base_0_rgb",
          "observation.images.left_wrist_camera": "observation.images.left_wrist_0_rgb",
          "observation.images.right_wrist_camera": "observation.images.right_wrist_0_rgb"}
TASK = "stack the three cubes into one tower"


def load(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    device = args.device
    if args.ckpt:
        ckpt = str(args.ckpt)
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = "cpu"
        adapter_cfg = Path(ckpt) / "adapter_config.json"
        if adapter_cfg.is_file():
            from peft import PeftModel
            base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
            if args.base_override:
                base = str(args.base_override)
            elif not Path(base).exists():
                raise SystemExit(f"adapter base '{base}' does not exist here; pass --base-override <patched pi05_base>")
            print(f"[server] LoRA checkpoint; base {base}; merging on CPU")
            policy = get_policy_class(cfg.type).from_pretrained(base, config=cfg, strict=False)
            policy = PeftModel.from_pretrained(policy, ckpt, torch_device="cpu").merge_and_unload()
        else:
            policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
        pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt,
                                             preprocessor_overrides={"device_processor": {"device": device, "float_dtype": args.dtype}})
        source = ckpt
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        base = str(args.base)
        meta = LeRobotDatasetMetadata(f"local/{Path(args.dataset_root).name}", root=args.dataset_root)
        cfg = PreTrainedConfig.from_pretrained(base)
        cfg.device = "cpu"
        cfg.empty_cameras = 0
        from lerobot.policies.factory import make_policy
        policy = make_policy(cfg=cfg, ds_meta=meta, rename_map=RENAME)
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=base, dataset_stats=meta.stats,
            preprocessor_overrides={"device_processor": {"device": device, "float_dtype": args.dtype},
                                    "normalizer_processor": {"stats": meta.stats, "features": {**cfg.input_features, **cfg.output_features}, "norm_map": cfg.normalization_mapping},
                                    "rename_observations_processor": {"rename_map": RENAME}},
            postprocessor_overrides={"unnormalizer_processor": {"stats": meta.stats, "features": cfg.output_features, "norm_map": cfg.normalization_mapping}})
        source = f"base:{base} stats:{args.dataset_root}"
    if args.dtype:
        policy = policy.to(getattr(torch, args.dtype))
    policy = policy.to(device).eval()
    print(f"[server] loaded {source}; chunk_size={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps} "
          f"device={device} dtype={args.dtype}; cuda mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB" if device.startswith("cuda") else "")
    return policy, pre, post


class Runner:
    def __init__(self, policy, pre, post, device: str, dtype: str | None):
        import torch
        self.t = torch
        self.policy, self.pre, self.post = policy, pre, post
        self.device = torch.device(device)
        self.autocast = getattr(torch, dtype) if dtype else None
        self.n_calls = 0

    def reset(self):
        self.policy.reset()

    def act(self, images: dict, state: np.ndarray, task: str) -> np.ndarray:
        torch = self.t
        obs = {}
        for k, img in images.items():
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            obs[f"observation.images.{k}"] = x.to(self.device)
        obs["observation.state"] = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0).to(self.device)
        obs["task"] = [task]
        cast = torch.autocast(self.device.type, dtype=self.autocast) if self.autocast else contextlib.nullcontext()
        with torch.inference_mode(), cast:
            obs = self.pre(obs)
            chunk = self.policy.predict_action_chunk(obs)
            chunk = self.post(chunk)
        self.n_calls += 1
        return np.asarray(chunk.squeeze(0).detach().float().cpu().numpy(), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path)
    g.add_argument("--base", type=Path)
    ap.add_argument("--dataset-root", type=Path, default=Path("data/fr3_duo_rand500"))
    ap.add_argument("--base-override", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--selftest", action="store_true", help="one inference on zeros, print the chunk, exit")
    a = ap.parse_args()
    policy, pre, post = load(a)
    runner = Runner(policy, pre, post, a.device, a.dtype)
    if a.selftest:
        imgs = {k.split(".")[-1]: np.zeros((224, 224, 3), np.uint8) for k in RENAME}
        st = np.zeros(16, np.float32)
        t0 = time.time()
        ch = runner.act(imgs, st, TASK)
        print(f"[selftest] chunk {ch.shape} in {time.time() - t0:.2f}s; row0 {ch[0].round(3)}")
        ch = runner.act(imgs, st, TASK)
        return 0
    with Listener(("127.0.0.1", a.port), authkey=AUTHKEY) as srv:
        print(f"[server] listening on 127.0.0.1:{a.port}", flush=True)
        while True:
            with srv.accept() as conn:
                print("[server] client connected", flush=True)
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    cmd = msg.get("cmd")
                    if cmd == "reset":
                        runner.reset()
                        conn.send("ok")
                    elif cmd == "act":
                        images, state = unpack_obs(msg["obs"])
                        t0 = time.time()
                        chunk = runner.act(images, state, msg.get("task", TASK))
                        conn.send({"chunk": pack(chunk), "ms": (time.time() - t0) * 1000})
                    elif cmd == "info":
                        conn.send({"chunk_size": int(policy.config.chunk_size), "calls": runner.n_calls,
                                   "source": str(a.ckpt or a.base), "dtype": a.dtype})
                    elif cmd == "quit":
                        conn.send("bye")
                        return 0
                    else:
                        conn.send({"error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    raise SystemExit(main())
