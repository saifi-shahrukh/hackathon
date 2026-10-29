"""Verify a pi0.5 LoRA checkpoint from the ARTIFACT, never from the flags.

A LoRA misconfiguration is invisible in the loss curve (the thesis lost ~4 GPU-days to it twice),
so the adapter file is counted: which parameter groups carry weights, how many, and whether the
openpi-parity settings (alpha == r, expert rank 32, six fully trained modules) are in the config.

    python fr3duo_pipeline/train/verify_adapter.py <run>/checkpoints/005000/pretrained_model
    python fr3duo_pipeline/train/verify_adapter.py <ckpt> --expect 450442000   # exact-total gate

Reference (thesis, patched pi05_base, same recipe, 2 cameras): 951 tensors / 450,442,000 learnable,
437 tensors in vision_tower. With three cameras the counts should be identical (the camera count
does not change the parameter set); any other total is a different recipe. Known bad totals:
1,385,984 = LeRobot's expert-only Q/V default (whole VLM frozen); 33.5 M = LoRA without the full
modules (the "blind" recipe). Treat a mismatch as a stop, not a note.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

GROUPS = ("language_model", "gemma_expert", "vision_tower", "multi_modal_projector",
          "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
FULL_EXPECTED = {"vision_tower", "multi_modal_projector", "action_in_proj", "action_out_proj",
                 "time_mlp_in", "time_mlp_out"}
BAD = {1_385_984: "LeRobot expert-only Q/V default: the whole VLM is frozen",
       33_506_304: "LoRA only, vision tower frozen (the thesis 'blind' recipe)",
       33_605_120: "LoRA only, vision tower frozen (pi0 variant)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="a .../pretrained_model directory")
    ap.add_argument("--expect", type=int, default=None, help="require this exact learnable total")
    a = ap.parse_args()
    from safetensors import safe_open

    adapter, cfgp = a.ckpt / "adapter_model.safetensors", a.ckpt / "adapter_config.json"
    if not adapter.is_file():
        print(f"[verify] FAIL: no adapter_model.safetensors under {a.ckpt} (not a LoRA run, or PEFT never engaged)")
        return 1
    cfg = json.loads(cfgp.read_text()) if cfgp.is_file() else {}
    r, alpha = cfg.get("r"), cfg.get("lora_alpha")
    rank_pat, alpha_pat = cfg.get("rank_pattern") or {}, cfg.get("alpha_pattern") or {}
    full = set(cfg.get("modules_to_save") or [])

    counts: Counter = Counter()
    total = 0
    lora_b_zero = 0
    lora_b = 0
    with safe_open(str(adapter), "pt") as g:
        for k in g.keys():
            t = g.get_tensor(k)
            total += t.numel()
            for grp in GROUPS:
                if grp in k:
                    counts[grp] += 1
            if "lora_B" in k:
                lora_b += 1
                lora_b_zero += int(t.count_nonzero().item() == 0)

    print(f"[verify] {a.ckpt}")
    print(f"  adapter_config : r={r} lora_alpha={alpha} rank_pattern={rank_pat} alpha_pattern={alpha_pat}")
    print(f"  full modules   : {sorted(full) or 'none'}")
    print(f"  tensors by grp : {dict(counts)}")
    print(f"  learnable total: {total:,}   lora_B all-zero: {lora_b_zero}/{lora_b}")

    problems = []
    if alpha != r:
        problems.append(f"lora_alpha={alpha} != r={r} (openpi parity needs alpha == r)")
    if not any(v == 32 for v in rank_pat.values()):
        problems.append("rank_pattern has no 32 entry (expert rank should be 32)")
    if full != FULL_EXPECTED:
        problems.append(f"modules_to_save {sorted(full)} != expected {sorted(FULL_EXPECTED)}")
    if counts["vision_tower"] == 0:
        problems.append("no vision_tower tensors in the adapter: the tower was NOT trained")
    if counts["language_model"] == 0 or counts["gemma_expert"] == 0:
        problems.append("LoRA missing on language_model or gemma_expert")
    if total in BAD:
        problems.append(f"total {total:,} matches a known-bad recipe: {BAD[total]}")
    if lora_b and lora_b_zero == lora_b:
        problems.append("every lora_B is zero: the adapter never trained (or is a fresh init)")
    if a.expect is not None and total != a.expect:
        problems.append(f"total {total:,} != --expect {a.expect:,}")
    if problems:
        print("[verify] FAIL:\n  - " + "\n  - ".join(problems))
        return 1
    print("[verify] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
