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
  fr3duo/clinic_scene.py       the native-data scene (default): plain tray-table in front of the robot, geometry and
                               cameras identical to rand500 (verified), looks: plain (eval) / rand500 / mixed (collection)
  fr3duo/scene.py              flat rand500-look floor scene used for the replay-fidelity checks
  fr3duo/expert.py             scripted IK stacking expert (dataset-like timing, binary gripper)
  assets/room/                 exported room meshes + clinic_scene.xml (tools/export_room_meshes.py, tools/build_room_mjcf.py)
  tools/, eval/                calibration, replay, generation, LeRobot conversion, policy server, evaluation
  tools/view_scene.py          MuJoCo viewer: --scene clinic|flat, --replay N
data/fr3_duo_native200/        the regenerated 200-episode MuJoCo dataset (LeRobot v3.0), committed
results/                       audits, replay fidelity, evaluation tables
```

## Pipeline on the workstation (RCS venv = MuJoCo side, pi0 venv = lerobot 0.4.4 side)

```bash
RCS=/home/robolab-2/ur5e_hande_workspace/robot-control-stack/.venv/bin/python
PI0=/home/robolab-2/ur5e_hande_workspace/Thesis/thesis/envs/pi0/bin/python
bash fr3duo_pipeline/tools/unpack_dataset.sh rand500                                  # data/fr3_duo_rand500
$PI0 fr3duo_pipeline/tools/audit_dataset.py data/fr3_duo_rand500                       # results/01_dataset_audit/fr3_duo_rand500_audit.json
$PI0 fr3duo_pipeline/tools/extract_keyframes.py data/fr3_duo_rand500 results/01_dataset_audit/rand500_frames.npz
MUJOCO_GL=egl $RCS fr3duo_pipeline/tools/calibrate_scene.py results/01_dataset_audit/rand500_frames.npz   # configs/scene_calib.json
MUJOCO_GL=egl $RCS fr3duo_pipeline/tools/replay_episode.py results/01_dataset_audit/rand500_frames.npz fr3duo_pipeline/configs/scene_calib.json --episodes 0-19 --out results/replay
$PI0 fr3duo_pipeline/tools/side_by_side.py data/fr3_duo_rand500 results/replay --episodes 0,3
MUJOCO_GL=egl $RCS fr3duo_pipeline/tools/gen_native.py --scene clinic --look mixed --n 200 --seed0 10000 --out data/native_raw --workers 8
$PI0 fr3duo_pipeline/tools/raw_to_lerobot.py data/native_raw data/fr3_duo_native200 --stats-from data/fr3_duo_rand500 --n 200
$PI0 fr3duo_pipeline/tools/check_dataset.py data/fr3_duo_native200 data/fr3_duo_rand500
# evaluation (GPU): one server per checkpoint, the same 50 seeds for every condition
$PI0 fr3duo_pipeline/eval/policy_server.py --ckpt checkpoints/A_pi05_rand500_5k/checkpoints/005000/pretrained_model --base-override $HF_HOME/patched_bases/pi05_base &
MUJOCO_GL=egl $RCS fr3duo_pipeline/eval/run_eval.py --condition A_5k_nas10 --n-action-steps 10   # --look plain by default
MUJOCO_GL=egl $RCS fr3duo_pipeline/eval/run_eval.py --condition expert --expert
python fr3duo_pipeline/eval/summarize.py results/eval --compare B_native_nas10 C_control_nas10
```

Look at the scene before collecting or evaluating (camera window = policy input + overview; `--mode gui` = MuJoCo viewer):
```bash
$RCS fr3duo_pipeline/tools/view_scene.py --episodes 2            # --look plain|rand500|mixed, --replay N, --scene flat
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
