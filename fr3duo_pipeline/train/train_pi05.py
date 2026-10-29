"""pi0.5 LoRA training wrapper for lerobot 0.4.4 (FR3 Duo, absolute joint actions).

Usage (what scripts/hpc/train_pi05.sbatch runs):

    python -m fr3duo_pipeline.train.train_pi05 <any lerobot-train arguments>

It forwards every argument to `lerobot.scripts.lerobot_train.train` after installing a few
monkeypatches that lerobot 0.4.4 lacks. All of them are ported from the thesis repo's
`src/policies/tools/train_delta.py` (Saifi, 2026-09), minus the delta-action machinery: this
project trains ABSOLUTE joint targets, exactly as stored in the datasets.

What is patched and why (every patch fails loud, never silently no-ops):

1. openpi LoRA parity. LeRobot's PeftConfig never sets `lora_alpha`, so PEFT's default alpha=8
   applies and every LoRA update is half the strength of openpi's reference (alpha == rank).
   openpi also uses rank 32 on the action expert and 16 on the language tower. Set through
   PEFT's `rank_pattern` / `alpha_pattern`. Disable with OPENPI_LORA_PARITY=false.
2. Full-module training. `--peft.full_training_modules` (PEFT `modules_to_save`) is how the SigLIP
   tower, the projector and the action/time projections train outright, as openpi's freeze filter
   does. lerobot 0.4.4 does not actually enable gradient checkpointing inside the trainable tower
   copy, so HF's own `gradient_checkpointing_enable(use_reentrant=False)` is called on it, and an
   audit prints the trainable parameters by group and refuses the known misconfigurations.
3. PI_FROZEN_FP16=true stores every frozen fp32 weight as fp16 (embeddings and norms stay fp32)
   so the run fits a 32 GB V100 at 16 per rank. Trainable weights stay fp32 masters, which
   accelerate's fp16 GradScaler requires.
4. Continuing a LoRA checkpoint (stages B and C). With `--policy.path=<stage A ckpt>` lerobot 0.4.4
   loads the adapter through `PeftModel.from_pretrained(..., is_trainable=False)`: every adapter and
   full-module weight arrives FROZEN and the optimiser would train nothing. This wrapper re-enables
   `requires_grad` on the `lora_` and `modules_to_save` parameters, verifies the adapter weights
   really were loaded (fresh LoRA B matrices are all zeros), and runs the same checkpointing / fp16 /
   audit as a fresh wrap.
5. A `--rename_map` source key that is not a dataset feature is silently ignored by lerobot and the
   policy would train on an empty camera slot; checked against the dataset's meta/info.json.
6. Peak CUDA memory is printed at exit, so a smoke run tells you whether the full run fits.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

TAG = "[train-pi05]"
KEEP32 = ("patch_embedding", "position_embedding", "input_layernorm", "post_attention_layernorm",
          "model.norm", "layer_norm", "layernorm")


def _arg(argv: list[str], key: str) -> str | None:
    return next((a.split("=", 1)[1] for a in argv if a.startswith(key + "=")), None)


def _assert_rename_map_matches_dataset(argv: list[str]) -> None:
    raw, ds = _arg(argv, "--rename_map"), _arg(argv, "--dataset.root")
    if not raw or not ds:
        print(f"{TAG} rename_map check skipped (no --rename_map or --dataset.root)")
        return
    rmap = json.loads(raw)
    feats = set(json.loads((Path(ds) / "meta" / "info.json").read_text())["features"])
    missing = [k for k in rmap if k not in feats]
    if missing:
        raise SystemExit(f"{TAG} --rename_map source key(s) {missing} are NOT dataset features of "
                         f"{Path(ds).name}; lerobot would ignore them silently. Image features: "
                         f"{sorted(f for f in feats if 'image' in f)}")
    print(f"{TAG} rename_map OK: {len(rmap)} source key(s) present in {Path(ds).name}")


def _assert_quantile_stats(argv: list[str]) -> None:
    """pi0.5 normalises state/action with QUANTILES; without q01/q99 lerobot fails late."""
    ds = _arg(argv, "--dataset.root")
    if not ds:
        return
    stats = json.loads((Path(ds) / "meta" / "stats.json").read_text())
    for k in ("observation.state", "action"):
        if not {"q01", "q99"} <= set(stats.get(k, {})):
            raise SystemExit(f"{TAG} {Path(ds).name}: meta/stats.json['{k}'] lacks q01/q99 "
                             f"(pi0.5 QUANTILES normalisation needs them)")
    print(f"{TAG} quantile stats OK (q01/q99 present for state and action)")


def _install_openpi_lora_parity() -> None:
    if os.environ.get("OPENPI_LORA_PARITY", "true").lower() != "true":
        print(f"{TAG} OPENPI_LORA_PARITY=false: PEFT defaults (alpha=8) left alone")
        return
    from lerobot.policies.pretrained import PreTrainedPolicy

    expert_rank = int(os.environ.get("PI0_LORA_EXPERT_RANK", "32"))
    _orig = PreTrainedPolicy._build_peft_config

    def _patched(self, cli_overrides: dict):
        cfg = _orig(self, cli_overrides)
        r = getattr(cfg, "r", None)
        if r is None:
            return cfg
        cfg.lora_alpha = r
        cfg.rank_pattern = {**(getattr(cfg, "rank_pattern", None) or {}), ".*gemma_expert.*": expert_rank}
        cfg.alpha_pattern = {**(getattr(cfg, "alpha_pattern", None) or {}), ".*gemma_expert.*": expert_rank}
        print(f"{TAG} openpi LoRA parity: r={r} alpha={cfg.lora_alpha} (scaling 1.0); "
              f"gemma_expert rank/alpha={expert_rank}")
        return cfg

    PreTrainedPolicy._build_peft_config = _patched


def _finish_peft_model(pm, policy_cfg, source: str):
    """Shared post-processing for a freshly wrapped OR a loaded PeftModel."""
    import torch
    from transformers import PreTrainedModel
    try:
        from peft.utils.other import ModulesToSaveWrapper
    except ImportError:  # older peft
        from peft.utils import ModulesToSaveWrapper

    cfg = pm.peft_config[getattr(pm, "active_adapter", "default")]
    full = list(getattr(cfg, "modules_to_save", None) or [])
    n_ckpt = 0
    for name, mod in pm.named_modules():
        if isinstance(mod, ModulesToSaveWrapper):
            for sub in list(mod.modules_to_save.values()):
                if any(k.startswith("lora_") for k, _ in sub.named_parameters()):
                    raise SystemExit(f"{TAG} '{name}' is BOTH a LoRA target and a full module; "
                                     f"remove it from --peft.target_modules. Refusing.")
                for m in sub.modules():
                    if isinstance(m, PreTrainedModel) and getattr(m, "supports_gradient_checkpointing", False):
                        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        n_ckpt += 1
    if os.environ.get("PI_FROZEN_FP16", "false").lower() == "true":
        n16 = 0
        for n, p in pm.named_parameters():
            if not p.requires_grad and p.dtype == torch.float32 and not any(k in n for k in KEEP32):
                p.data = p.data.to(torch.float16)
                n16 += p.numel()
        print(f"{TAG} PI_FROZEN_FP16: {n16:,} frozen weights stored as fp16 (embeddings/norms kept fp32)")
    groups: dict[str, int] = {}
    for n, p in pm.named_parameters():
        if not p.requires_grad:
            continue
        g = ("vision_tower" if "vision_tower" in n else "multi_modal_projector" if "multi_modal_projector" in n
             else "language_model" if "language_model" in n else "gemma_expert" if "gemma_expert" in n
             else "projections")
        groups[g] = groups.get(g, 0) + p.numel()
        if p.dtype != torch.float32 and os.environ.get("ACCELERATE_MIXED_PRECISION", "").lower() == "fp16":
            raise SystemExit(f"{TAG} trainable '{n}' is {p.dtype}: accelerate's fp16 GradScaler refuses "
                             f"non-fp32 grads (use --policy.dtype=float32 on the V100)")
    total = sum(groups.values())
    print(f"{TAG} [{source}] trainable by group: " + ", ".join(f"{k} {v:,}" for k, v in sorted(groups.items()))
          + f" | total {total:,} | full modules {full or 'none'} | HF grad-ckpt on {n_ckpt} module(s)")
    if total == 0:
        raise SystemExit(f"{TAG} NO trainable parameters. Refusing.")
    for f in full:
        if not any(f in n for n, p in pm.named_parameters() if p.requires_grad):
            raise SystemExit(f"{TAG} full_training_modules entry '{f}' matched no trainable parameter. Refusing.")
    if "vision_tower" in full and n_ckpt == 0:
        raise SystemExit(f"{TAG} vision_tower is fully trained but no HF gradient checkpointing engaged. Refusing.")
    frozen_flag = getattr(policy_cfg, "freeze_vision_encoder", False) or any(
        getattr(m, "freeze_vision_encoder", False) is True for m in pm.modules())
    if "vision_tower" in full and frozen_flag:
        raise SystemExit(f"{TAG} vision_tower is a full module but freeze_vision_encoder=True; pass "
                         f"--policy.freeze_vision_encoder=false")
    if n_ckpt:
        pm.train()
        idle = [n for n, m in pm.named_modules()
                if "modules_to_save" in n and getattr(m, "gradient_checkpointing", False) and not m.training]
        if idle:
            raise SystemExit(f"{TAG} gradient checkpointing enabled but INACTIVE on {len(idle)} module(s), "
                             f"e.g. '{idle[0]}'. Refusing.")
    return pm


def _install_full_module_training() -> None:
    from lerobot.policies.pretrained import PreTrainedPolicy
    _orig = PreTrainedPolicy.wrap_with_peft

    def _patched(self, peft_config=None, peft_cli_overrides=None):
        pm = _orig(self, peft_config=peft_config, peft_cli_overrides=peft_cli_overrides)
        return _finish_peft_model(pm, self.config, "fresh LoRA wrap")

    PreTrainedPolicy.wrap_with_peft = _patched


def _install_continue_from_adapter(T) -> None:
    """Stages B/C: `--policy.path` points at a stage-A LoRA checkpoint and no --peft.* is given."""
    import torch
    _orig = T.make_policy

    def _patched(*args, **kwargs):
        policy = _orig(*args, **kwargs)
        try:
            from peft import PeftModel
        except ImportError:
            return policy
        if not isinstance(policy, PeftModel):
            return policy
        n_on, n_b, n_b_nonzero = 0, 0, 0
        for n, p in policy.named_parameters():
            if "lora_" in n or "modules_to_save" in n:
                p.requires_grad_(True)
                n_on += 1
                if "lora_B" in n:
                    n_b += 1
                    n_b_nonzero += int(torch.count_nonzero(p.detach()).item() > 0)
        if n_b and n_b_nonzero == 0:
            raise SystemExit(f"{TAG} continuing from an adapter whose lora_B weights are ALL ZERO: the "
                             f"checkpoint was not loaded (fresh init). Refusing.")
        print(f"{TAG} continue-from-adapter: re-enabled grads on {n_on} adapter/full-module tensors; "
              f"lora_B non-zero in {n_b_nonzero}/{n_b} matrices (checkpoint weights present)")
        cfg = getattr(policy, "config", None)
        if cfg is not None and getattr(cfg, "use_peft", False) is not True:
            cfg.use_peft = True
        return _finish_peft_model(policy, cfg, "loaded adapter")

    T.make_policy = _patched


def _install_peak_mem_report() -> None:
    def _report() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2**30
                tot = torch.cuda.get_device_properties(0).total_memory / 2**30
                print(f"{TAG} CUDA PEAK {peak:.1f} GiB of {tot:.1f} GiB ({100 * peak / tot:.0f} %)")
        except Exception as e:  # noqa: BLE001
            print(f"{TAG} peak memory report failed: {type(e).__name__}: {e}")
    atexit.register(_report)


def _warn_video_decoder() -> None:
    """The thesis venvs carry a direct-pyav decoder patch with thread_count=1 (an AV1 decoder
    thread explosion deadlocked multi-GPU runs). Warn when it is absent."""
    try:
        import lerobot.datasets.video_utils as vu
        src = Path(vu.__file__).read_text()
        if "THESIS PYAV PATCH" not in src:
            print(f"{TAG} WARNING: lerobot video_utils has no single-thread pyav patch; with DDP, AV1 "
                  f"decoding may spawn thousands of threads. Keep --num_workers small (2-4).")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    argv = sys.argv
    _assert_rename_map_matches_dataset(argv)
    _assert_quantile_stats(argv)
    _warn_video_decoder()

    import lerobot.scripts.lerobot_train as T

    _install_openpi_lora_parity()
    _install_full_module_training()
    _install_continue_from_adapter(T)
    _install_peak_mem_report()
    print(f"{TAG} absolute joint actions (no delta step); forwarding to lerobot 0.4.4 train()")
    T.train()


if __name__ == "__main__":
    main()
