# FR3 Duo cube stacking: pi0.5 "native MuJoCo data" replication

Siemens Healthineers 2026 Hackathon, Challenge C2 "Teach the robot". This repository rebuilds,
from zero, the pipeline behind the pitched result and tests it honestly:

| Claim from the pitch | Status |
|---|---|
| pi0.5 LoRA, 5k steps on the organizer set `fr3_duo_rand500` → 0 % success in MuJoCo | to be measured (`RESULTS.md`) |
| + 200 scripted-expert episodes recorded in the same MuJoCo env, 2.5k more steps → 54 % | to be measured |
| the gain comes from the native data (control: 2.5k more steps on rand500 instead) | to be measured |

The original hackathon code and data were deleted, so nothing here is a copy; every number in
`RESULTS.md` is produced by the scripts in `fr3duo_pipeline/` and committed JSON under `results/`.

## Layout

```
fr3duo_pipeline/
  train/train_pi05.py          lerobot 0.4.4 wrapper: openpi-parity LoRA, full-module training, fp16 frozen weights
  train/verify_adapter.py      counts a saved adapter (gate before trusting a checkpoint)
  scripts/hpc/train_pi05.sbatch  SLURM launcher for stages A / B / C (IIS HPC, thesis venv)
  scripts/hpc/preflight.sh     health board before sbatch
  tools/unpack_dataset.sh      unzip the organizer data once into data/
  fr3duo/, tools/, eval/       MuJoCo scene, calibration, expert, native dataset, evaluation (in progress)
data/fr3_duo_native200/        the regenerated 200-episode MuJoCo dataset (LeRobot v3.0), committed
results/                       audits, replay fidelity, evaluation tables
```

## Training on the HPC (stage A first)

```bash
git clone https://github.com/saifi-shahrukh/hackathon.git ~/hackathon && cd ~/hackathon
# copy C2_Teach-the-robot_VLA/Dataset/fr3_duo_rand500.zip here by hand (2.1 GB), then:
bash fr3duo_pipeline/tools/unpack_dataset.sh rand500
bash fr3duo_pipeline/scripts/hpc/preflight.sh A            # must print GREEN
SLURM_SUBMIT_DIR=$PWD SLURM_JOB_GPUS=0,1 SLURM_CPUS_PER_TASK=16 DRY_RUN=1 \
  bash fr3duo_pipeline/scripts/hpc/train_pi05.sbatch A      # prints the exact command
sbatch --partition=gpu-short --time=01:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=40G \
  --job-name=fr3duo_pi05_smoke --export=ALL,STEPS=30,SAVE_FREQ=30,BATCH=16,RUN_TAG=_smoke \
  fr3duo_pipeline/scripts/hpc/train_pi05.sbatch A           # 30-step smoke: read CUDA PEAK
sbatch fr3duo_pipeline/scripts/hpc/train_pi05.sbatch A      # 5000 steps, 2 GPUs, ckpt at 2500/5000
```
Stages B (native data) and C (control) run the same way once stage A has `checkpoints/005000`
and, for B, `data/fr3_duo_native200` exists (it comes with `git pull`). Details and knobs are in
the header of `train_pi05.sbatch`. Checkpoints are fetched back to the workstation for evaluation.

Requirements on the cluster: the thesis repo at `~/thesis` with its `scripts/hpc/.venv`
(lerobot 0.4.4, torch cu118, peft) and the patched `pi05_base` under `$HF_HOME/patched_bases/`.

## Recipe (what trains)

pi0.5 (`lerobot/pi05_base`), absolute joint targets (16-D: 7 joints + gripper per arm), three
224×224 cameras mapped to `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`, chunk 50,
QUANTILES normalisation from the dataset stats. LoRA r=16 (alpha=r) on the Gemma language tower
and action expert (expert rank 32), with the SigLIP vision tower, multimodal projector and the
action/time projections trained outright; gradient checkpointing; fp16 autocast with fp32 master
weights and fp16-stored frozen weights (V100, no bf16); AdamW 2.5e-5, cosine with 1000 warm-up
steps; batch 16 per rank; seed 1000. The hackathon run's exact settings are not recorded, so these
follow the openpi reference recipe and are stated as assumptions in `RESULTS.md`.
