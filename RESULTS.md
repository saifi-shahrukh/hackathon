# RESULTS: does the pitched "native MuJoCo data" result hold?

Status: **in progress** (2026-10-29). Everything below is measured by the scripts in
`fr3duo_pipeline/` and stored under `results/`; nothing is taken from memory of the hackathon.
Unmeasured items are marked **pending**.

## 1. The claim (Siemens Healthineers 2026 Hackathon, Challenge C2, pitch of 2026-10-24)

pi0.5 (LeRobot PyTorch, LoRA, absolute joint actions) fine-tuned 5k steps on the organizer dataset
`fr3_duo_rand500` scored **0 %** on the cube-stacking task in a MuJoCo environment built to match the
dataset. 200 episodes recorded with a scripted/IK expert in that MuJoCo environment, used to
fine-tune the 5k checkpoint for 2.5k more steps, raised success to **54 %**. Success = all three
cubes stacked. The original code, environment, expert data and checkpoints were deleted after the
event, so this is a from-zero rebuild. A control arm (2.5k more steps on rand500 instead of the
native data, same compute) is added to test whether the gain is due to the native data.

## 2. What was rebuilt vs assumed

| Element | Status | Evidence |
|---|---|---|
| Dataset facts (6 fps, 490 eps, 16-D absolute joints, binary gripper 1 = open, one close/open per arm, all episodes start at the RCS home pose) | measured | `results/01_dataset_audit/fr3_duo_rand500_audit.json` |
| Simulator identity: rand500 was rendered with robot-control-stack's `EmptyWorldFR3Duo` MuJoCo scene (not Isaac Sim) | strong evidence | frame-0 home pose equals `rcs.HOME_POSITIONS` to 1e-6 rad; all three camera views match (`results/02_scene_replay/episodes_00-09/side_by_side_ep000.png`) |
| Cube size 4.2 cm, cubes on the floor plane (z = 0) | measured | FK at grasp frames: TCP 3.74 cm ± 3 mm; gripper reading 0.521 when holding a 4.2 cm cube vs 0.522 in the data (`fr3duo_pipeline/configs/scene_calib.json`) |
| Spawn rule: x in [0.35, 0.65], y in [-0.15, 0.15]; base cube random; left arm takes the larger-y of the other two | inferred from data | B_y > C_y in 490/490 episodes; only the three y-orders this rule allows occur |
| Table texture / mat colour / lighting | approximated | procedural wood planks, random tint, flat mat, random light; the organizers' textures are not available |
| Replay fidelity: recorded commands rebuild the tower in our scene | **19 / 20 episodes (95 %)** | `results/02_scene_replay/episodes_00-09/summary.json`, `results/02_scene_replay/episodes_10-19/summary.json` |
| Scripted expert (IK, dataset-like timing) | re-implemented | 6/6 on the first seeds; gripper events at 43–45 / 109–111 / 214–221 / 285–290 vs data medians 43 / 109 / 215 / 288; hold reading 0.522; peak joint speed ≤ 0.61 rad/s (data ≤ 0.74) |
| 200 native episodes (LeRobot v3.0, same schema, AV1) | **pending** (generating) | `data/fr3_duo_native200/`, `results/native_vs_rand500/check.json` |
| pi0.5 recipe (LoRA r16 alpha=r, expert r32, SigLIP + projector + action/time projections trained, grad ckpt, fp16 autocast on V100, chunk 50, batch 16/rank, lr 2.5e-5 cosine) | assumed = openpi reference recipe; the hackathon run's exact settings are not recorded | `fr3duo_pipeline/train/train_pi05.py`, `scripts/hpc/train_pi05.sbatch` |
| lerobot 0.4.4 (HPC venv) for training and serving | by choice (the HPC venv) | the dataset itself was written with lerobot 0.6.1; it loads unchanged in 0.4.4 (Gate 0) |

## 3. Replay fidelity (scene vs generator)

Open-loop replay of each episode's recorded joint + gripper commands at 6 Hz, cubes placed at the
positions derived from forward kinematics:

| episodes | tower rebuilt | joint tracking p95 |
|---|---|---|
| 0–9 | 10 / 10 | 0.052–0.063 rad |
| 10–19 | 9 / 10 | similar |

A 95 % replay rate bounds what the scene can explain: if a policy trained on rand500 fails here,
at most ~5 % of that can be attributed to geometry mismatch. Visual mismatch (textures, lighting)
is a separate, unquantified factor and is listed as a limitation.

## 4. Expert and native dataset

**pending**: expert success rate over all attempts (`data/native_raw/generation_log.jsonl`),
dataset parity check, side-by-side frames.

## 5. Training runs

| run | init | data | steps | batch | GPUs | wall time | loss | checkpoint |
|---|---|---|---|---|---|---|---|---|
| A | patched pi05_base | rand500 | 5000 | 16/rank | | **pending** | | |
| B | A@5000 | native200 | +2500 | 16/rank | | **pending** | | |
| C (control) | A@5000 | rand500 | +2500 | 16/rank | | **pending** | | |

## 6. Evaluation (50 shared seeds, 360-frame cap, 6 Hz, chunked execution)

**pending**: `results/eval_table.md` (expert, base, A, B, C; n_action_steps 10 primary, 50 secondary).

## 7. Replication status

| Claim element | Status |
|---|---|
| 0 % after 5k steps on rand500 | **pending** |
| 54 % after +2.5k steps on 200 native episodes | **pending** |
| The gain is due to the native data (B vs control C) | **pending** |

## 8. Known limitations

- The organizers' generator, textures and lighting are not available; the scene is a geometric
  reconstruction with an approximated look. Domain gap between rand500 images and this scene is
  real and affects every policy condition equally, but it can lower "A" below what the organizers'
  own environment would give.
- The hackathon's exact training settings (chunk size, precision, LoRA ranks, steps/batch) are
  not recorded; openpi reference settings are used and stated.
- The hackathon's `n_action_steps` at evaluation is unknown; two settings are reported.

## 9. Reproduce

See `README.md`. Commits and dataset md5s are recorded next to each result file.
