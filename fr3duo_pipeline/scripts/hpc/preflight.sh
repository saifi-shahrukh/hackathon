#!/bin/bash
# Pre-submission health board for train_pi05.sbatch on the IIS HPC headnode.
#   bash fr3duo_pipeline/scripts/hpc/preflight.sh <A|B|C>
# Green = safe to sbatch. Every check names what to fix.
set -uo pipefail
STAGE="${1:-A}"
HACK_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
THESIS_ROOT="${THESIS_ROOT:-$HOME/thesis}"
ok=0; bad=0
pass() { echo "  [ok]   $*"; ok=$((ok+1)); }
fail() { echo "  [FAIL] $*"; bad=$((bad+1)); }

echo "fr3duo pi0.5 preflight  stage=$STAGE  repo=$HACK_ROOT  thesis=$THESIS_ROOT"
if [ -f "$THESIS_ROOT/scripts/hpc/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$THESIS_ROOT/scripts/hpc/env.sh"; pass "thesis env.sh sourced (HF_HOME=$HF_HOME)"
else
  fail "no $THESIS_ROOT/scripts/hpc/env.sh (set THESIS_ROOT to the thesis clone)"; HPC_PY=""; HF_HOME="${HF_HOME:-}"
fi

if [ -n "${HPC_PY:-}" ] && [ -x "$HPC_PY" ]; then
  LDP=""; [ -d "${HPC_SYSROOT:-/nonexistent}" ] && LDP="$HPC_SYSROOT/lib64:$HPC_SYSROOT/usr/lib64:$HPC_VENV/lib"
  out="$(LD_LIBRARY_PATH="$LDP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH="$HACK_ROOT" "$HPC_PY" - <<'PY' 2>&1
import lerobot, peft, transformers, torch, accelerate, av
print(f"lerobot {lerobot.__version__} peft {peft.__version__} transformers {transformers.__version__} torch {torch.__version__} accelerate {accelerate.__version__} av {av.__version__}")
assert lerobot.__version__.startswith("0.4."), "expected lerobot 0.4.x (the thesis venv)"
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa
import fr3duo_pipeline.train.train_pi05  # noqa
print("imports-ok")
PY
)"
  if echo "$out" | grep -q imports-ok; then pass "venv imports: $(echo "$out" | head -1)"; else fail "venv import problem: $out"; fi
else
  fail "thesis venv python not executable (${HPC_PY:-unset}); run thesis bootstrap_hpc_venv.sh"
fi

BASE="${PI05_BASE:-${HF_HOME:-/nonexistent}/patched_bases/pi05_base}"
if [ -f "$BASE/config.json" ] && [ -f "$BASE/policy_preprocessor.json" ]; then
  if grep -q relative_actions_processor "$BASE/policy_preprocessor.json"; then
    fail "patched base still lists relative_actions_processor (lerobot 0.4.4 cannot load it): rerun thesis prefetch_bases.sh"
  else
    pass "patched pi05_base at $BASE"
  fi
else
  fail "patched pi05_base missing at $BASE (thesis scripts/hpc/prefetch_bases.sh)"
fi
TOK="${HF_HOME:-/nonexistent}/hub/models--google--paligemma-3b-pt-224"
[ -d "$TOK" ] && pass "PaliGemma tokenizer cached" || fail "PaliGemma tokenizer not cached (hf auth login; prefetch_bases.sh)"

DATA_DIR="${DATA_DIR:-$HACK_ROOT/data}"
case "$STAGE" in B) DS=fr3_duo_native200 ;; *) DS=fr3_duo_rand500 ;; esac
if [ -f "$DATA_DIR/$DS/meta/info.json" ]; then
  n=$(python3 -c "import json;d=json.load(open('$DATA_DIR/$DS/meta/info.json'));print(d['total_episodes'],d['fps'],d['codebase_version'])" 2>/dev/null || echo "?")
  pass "dataset $DS present (episodes fps version: $n)"
  python3 -c "import json;s=json.load(open('$DATA_DIR/$DS/meta/stats.json'));assert 'q01' in s['action'] and 'q99' in s['observation.state']" 2>/dev/null \
    && pass "q01/q99 stats present" || fail "meta/stats.json lacks q01/q99 for action/state"
  nvid=$(find "$DATA_DIR/$DS/videos" -name '*.mp4' 2>/dev/null | wc -l); [ "$nvid" -gt 0 ] && pass "$nvid video files" || fail "no videos under $DATA_DIR/$DS/videos"
else
  fail "dataset missing: $DATA_DIR/$DS (A/C: unpack_dataset.sh; B: git pull brings data/fr3_duo_native200)"
fi
if [ "$STAGE" != "A" ]; then
  INIT="${INIT:-$HACK_ROOT/checkpoints/A_pi05_rand500_5k/checkpoints/005000/pretrained_model}"
  [ -f "$INIT/adapter_config.json" ] && pass "stage-A checkpoint $INIT" || fail "stage-A LoRA checkpoint missing at $INIT (run stage A first)"
fi
command -v sbatch >/dev/null && pass "sbatch available ($(sinfo -h -o '%P' 2>/dev/null | tr '\n' ' '))" || fail "sbatch not in PATH (run on the headnode)"
[ -d "/data/beegfs/home/$USER" ] && pass "BeeGFS home present" || fail "/data/beegfs/home/$USER missing (sbatch --output path)"

echo
if [ "$bad" -eq 0 ]; then
  echo "GREEN ($ok checks). Dry run next:"
  echo "  SLURM_SUBMIT_DIR=\$PWD SLURM_JOB_GPUS=0,1 SLURM_CPUS_PER_TASK=16 DRY_RUN=1 bash fr3duo_pipeline/scripts/hpc/train_pi05.sbatch $STAGE"
else
  echo "RED: $bad failing check(s), $ok ok. Fix them before sbatch."; exit 1
fi
