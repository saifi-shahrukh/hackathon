#!/bin/bash
#SBATCH --job-name=fr3duo_pi05
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=infinite
#SBATCH --output=/data/beegfs/home/%u/logs/%x_%j.out
#
# pi0.5 LoRA fine-tuning for the FR3 Duo cube-stacking replication, on the IIS HPC, inside the
# thesis venv (~/thesis/scripts/hpc/.venv: lerobot 0.4.4, torch cu118, peft, patched pi05_base).
#
#   usage:  sbatch fr3duo_pipeline/scripts/hpc/train_pi05.sbatch <A|B|C>      (from the repo root)
#
#   A  patched lerobot/pi05_base  + 5000 steps on data/fr3_duo_rand500      (LoRA, openpi parity)
#   B  A@5000                     + 2500 steps on data/fr3_duo_native200    (the pitched recipe)
#   C  A@5000                     + 2500 steps on data/fr3_duo_rand500      (control: same compute)
#
#   Pre-reqs on the headnode (once):
#     git clone https://github.com/saifi-shahrukh/hackathon.git ~/hackathon && cd ~/hackathon
#     ls ~/thesis/scripts/hpc/.venv/bin/python3.11          # the thesis venv (bootstrap_hpc_venv.sh)
#     ls $HF_HOME/patched_bases/pi05_base                   # thesis prefetch_bases.sh output
#     copy fr3_duo_rand500.zip (+ .md5) to ~/hackathon/C2_Teach-the-robot_VLA/Dataset/ and run
#       bash fr3duo_pipeline/tools/unpack_dataset.sh       # -> data/fr3_duo_rand500
#     (for B: data/fr3_duo_native200 is inside the git repo)
#     bash fr3duo_pipeline/scripts/hpc/preflight.sh A       # must be green
#
#   Smoke first (one GPU, gpu-short, 30 steps; read "trainable by group" and "CUDA PEAK"):
#     sbatch --partition=gpu-short --time=01:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=40G \
#            --job-name=fr3duo_pi05_smoke --export=ALL,STEPS=30,SAVE_FREQ=30,BATCH=16,RUN_TAG=_smoke \
#            fr3duo_pipeline/scripts/hpc/train_pi05.sbatch A
#     python fr3duo_pipeline/train/verify_adapter.py checkpoints/A_pi05_rand500_5k_smoke/checkpoints/000030/pretrained_model
#
#   Reference launch (2 GPUs, DDP, fp16 autocast, 16 per rank):
#     sbatch fr3duo_pipeline/scripts/hpc/train_pi05.sbatch A
#     sbatch fr3duo_pipeline/scripts/hpc/train_pi05.sbatch B      # after A has checkpoints/005000
#     sbatch fr3duo_pipeline/scripts/hpc/train_pi05.sbatch C
#   One whole node: sbatch --gres=gpu:4 --cpus-per-task=32 --mem=160G --export=ALL,BATCH=64 ... A
#   (BATCH is the TOTAL batch; per-rank stays 16. Memory per GPU, not GPU count, is the limit.)
#
#   Dry run on the headnode (prints the exact command, trains nothing):
#     SLURM_SUBMIT_DIR=$PWD SLURM_JOB_GPUS=0,1 SLURM_CPUS_PER_TASK=16 DRY_RUN=1 \
#       bash fr3duo_pipeline/scripts/hpc/train_pi05.sbatch A
#
#   Knobs (env, --export=ALL,...): STEPS SAVE_FREQ BATCH SEED RUN_TAG DATA_DIR THESIS_ROOT
#     PI_FROZEN_FP16 (default true) OPENPI_LORA_PARITY (default true) NUM_WORKERS (default 4)
#     INIT (override the checkpoint a stage starts from) WANDB=true (else disabled)
#   A finished run is never overwritten: an existing output dir with checkpoints/last resumes it.
set -euo pipefail

STAGE="${1:?usage: sbatch train_pi05.sbatch <A|B|C>}"
export DRY_RUN="${DRY_RUN:-0}"
HACK_ROOT="${HACK_ROOT:-${SLURM_SUBMIT_DIR:?run via sbatch from the repo root (dry run: SLURM_SUBMIT_DIR=\$PWD)}}"
cd "$HACK_ROOT"
THESIS_ROOT="${THESIS_ROOT:-$HOME/thesis}"
[ -f "$THESIS_ROOT/scripts/hpc/env.sh" ] || { echo "ERROR: $THESIS_ROOT/scripts/hpc/env.sh not found (set THESIS_ROOT)" >&2; exit 2; }
# shellcheck disable=SC1091
source "$THESIS_ROOT/scripts/hpc/env.sh"
hpc_offline_hf                      # HF_HUB_OFFLINE=1: compute nodes are air-gapped
[ -x "$HPC_PY" ] || { echo "ERROR: thesis venv python missing at $HPC_PY" >&2; exit 2; }
PY_LD=()
[ -d "${HPC_SYSROOT:-/nonexistent}" ] && PY_LD=(env "LD_LIBRARY_PATH=$HPC_SYSROOT/lib64:$HPC_SYSROOT/usr/lib64:$HPC_VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
export PYTHONPATH="$HACK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── resources: what SLURM allocated is what trains ─────────────────────────────────────────────
NPROC="$(echo "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}" | tr ',' '\n' | grep -c . || true)"
[ "${NPROC:-0}" -ge 1 ] || NPROC=1
BATCH="${BATCH:-$(( 16 * NPROC ))}"
[ $(( BATCH % NPROC )) -eq 0 ] || { echo "ERROR: BATCH=$BATCH not divisible by $NPROC GPUs" >&2; exit 2; }
PER_RANK=$(( BATCH / NPROC ))
MIXED_PRECISION="${MIXED_PRECISION:-${HPC_MIXED_PRECISION:-fp16}}"   # V100 = sm_70: fp16, never bf16
_OMP=$(( ${SLURM_CPUS_PER_TASK:-8} / NPROC )); [ "$_OMP" -ge 1 ] || _OMP=1
export OMP_NUM_THREADS="$_OMP" MKL_NUM_THREADS="$_OMP"
ulimit -n 131072 2>/dev/null || [ "$DRY_RUN" = "1" ] || echo "WARNING: could not raise ulimit -n"
export PI_FROZEN_FP16="${PI_FROZEN_FP16:-true}"
export OPENPI_LORA_PARITY="${OPENPI_LORA_PARITY:-true}"
export TOKENIZERS_PARALLELISM=false

# ── stage table ────────────────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$HACK_ROOT/data}"
PI05_BASE="${PI05_BASE:-$HF_HOME/patched_bases/pi05_base}"
A_OUT="$HACK_ROOT/checkpoints/A_pi05_rand500_5k"
case "$STAGE" in
  A) DS=fr3_duo_rand500;    STEPS="${STEPS:-5000}"; INIT="${INIT:-$PI05_BASE}";                                 PEFT=1; RUN=A_pi05_rand500_5k ;;
  B) DS=fr3_duo_native200;  STEPS="${STEPS:-2500}"; INIT="${INIT:-$A_OUT/checkpoints/005000/pretrained_model}"; PEFT=0; RUN=B_pi05_A5k_native200_2p5k ;;
  C) DS=fr3_duo_rand500;    STEPS="${STEPS:-2500}"; INIT="${INIT:-$A_OUT/checkpoints/005000/pretrained_model}"; PEFT=0; RUN=C_pi05_A5k_rand500_2p5k_control ;;
  *) echo "ERROR: stage must be A, B or C (got '$STAGE')" >&2; exit 2 ;;
esac
RUN="${RUN}${RUN_TAG:-}"
OUT="$HACK_ROOT/checkpoints/$RUN"
SAVE_FREQ="${SAVE_FREQ:-2500}"
SEED="${SEED:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
[ -d "$DATA_DIR/$DS/meta" ] || { echo "ERROR: dataset $DATA_DIR/$DS missing (meta/). Unpack or copy it first." >&2; exit 2; }
[ -f "$INIT/config.json" ] || { echo "ERROR: init checkpoint/base $INIT has no config.json" >&2; exit 2; }
if [ "$STAGE" != "A" ] && [ ! -f "$INIT/adapter_config.json" ]; then
  echo "ERROR: stage $STAGE must start from a stage-A LoRA checkpoint (adapter_config.json missing in $INIT)" >&2; exit 2
fi

RENAME='{"observation.images.external_camera":"observation.images.base_0_rgb","observation.images.left_wrist_camera":"observation.images.left_wrist_0_rgb","observation.images.right_wrist_camera":"observation.images.right_wrist_0_rgb"}'
# openpi-parity LoRA: r16 on Gemma + expert (expert rank/alpha raised to 32 by train_pi05.py),
# the SigLIP tower / projector / action+time projections trained outright (PEFT modules_to_save).
LORA_TARGETS='(.*\.(gemma_expert|language_model)\..*\.self_attn\.(q|k|v|o)_proj|.*\.(gemma_expert|language_model)\..*\.mlp\.(gate|up|down)_proj)'
FULL_MODULES='["vision_tower","multi_modal_projector","action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'

PEFT_ARGS=()
[ "$PEFT" = "1" ] && PEFT_ARGS=(--peft.r=16 --peft.target_modules="$LORA_TARGETS" --peft.full_training_modules="$FULL_MODULES")
WANDB_ARGS=(--wandb.enable=false)
if [ "${WANDB:-false}" = "true" ]; then hpc_wandb_setup; WANDB_ARGS=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-fr3duo_pi05}" --wandb.disable_artifact=true); fi

if [ "$NPROC" -gt 1 ]; then
  export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}" TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
  LAUNCH=("${PY_LD[@]}" "$(dirname "$HPC_PY")/accelerate" launch --multi_gpu --num_processes="$NPROC" --mixed_precision="$MIXED_PRECISION" --dynamo_backend=no -m)
else
  export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-$MIXED_PRECISION}"
  LAUNCH=("${PY_LD[@]}" "$HPC_PY" -m)
fi

if [ -e "$OUT/checkpoints/last" ]; then
  echo "[stage $STAGE] $OUT already has checkpoints/last -> RESUMING (set RUN_TAG for a fresh run)"
  CMD=("${LAUNCH[@]}" fr3duo_pipeline.train.train_pi05
       --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true --steps="$STEPS")
else
  CMD=("${LAUNCH[@]}" fr3duo_pipeline.train.train_pi05
       --dataset.repo_id="local/$DS" --dataset.root="$DATA_DIR/$DS" --dataset.video_backend=pyav
       --policy.path="$INIT" --policy.device=cuda --policy.dtype=float32
       --policy.push_to_hub=false --policy.repo_id="local/pi05_$RUN"
       --policy.chunk_size=50 --policy.n_action_steps=50 --policy.empty_cameras=0
       --policy.freeze_vision_encoder=false --policy.gradient_checkpointing=true
       --rename_map="$RENAME"
       ${PEFT_ARGS[@]+"${PEFT_ARGS[@]}"}
       --output_dir="$OUT" --job_name="$RUN"
       --batch_size="$PER_RANK" --steps="$STEPS" --save_freq="$SAVE_FREQ" --seed="$SEED"
       --num_workers="$NUM_WORKERS" --log_freq=50
       "${WANDB_ARGS[@]}")
fi

hpc_banner "fr3duo pi0.5  stage=$STAGE  run=$RUN  job=${SLURM_JOB_ID:-local}  node=${SLURMD_NODENAME:-$(hostname)}$([ "$DRY_RUN" = "1" ] && echo '  <-- DRY RUN')" \
  "dataset=$DATA_DIR/$DS  init=$INIT" \
  "GPUs=$NPROC  batch=$BATCH total = $PER_RANK/rank  precision=$MIXED_PRECISION autocast, fp32 masters, frozen fp16=$PI_FROZEN_FP16" \
  "steps=$STEPS  save_freq=$SAVE_FREQ  seed=$SEED  LoRA=$([ "$PEFT" = 1 ] && echo 'r16 parity (fresh)' || echo 'continue loaded adapter')  out=$OUT"
printf '  %q' "${CMD[@]}"; echo
[ "$DRY_RUN" = "1" ] && { echo "[dry run] nothing trained"; exit 0; }
mkdir -p "$HACK_ROOT/checkpoints" "$(dirname "/data/beegfs/home/$USER/logs/x")"
git -C "$HACK_ROOT" rev-parse HEAD > "$HACK_ROOT/checkpoints/${RUN}.commit" 2>/dev/null || true
srun "${CMD[@]}"
