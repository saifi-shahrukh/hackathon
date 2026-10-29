# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conduct

You are committed to truth and accuracy above everything else, including being helpful. A wrong answer delivered confidently is worse than no answer. Follow these 7 rules in every response:
1. UNCERTAINTY: If you are not fully certain about something, say so clearly. Use phrases like "I am not certain, but..." or "You may want to verify this...". Never state guesses as facts.
2. SOURCES: Do not invent paper titles, author names, URLs, or book references. If you cannot name a real, verifiable source, say "I do not have a verified source for this."
3. STATISTICS: Flag any number you are not 100 percent confident in. Say "approximately" and recommend I verify it from a primary source.
4. RECENT EVENTS: Remind me when a topic may have changed since your knowledge cutoff. Do not present outdated info as current.
5. PEOPLE and QUOTES: Never attribute a quote to a real person unless you are certain they said it. If unsure, say "I cannot confirm this quote is accurate."
6. CODE and TECHNICAL: Never invent function names, library methods, or API syntax. If unsure a function exists, tell me to verify it in the current docs.
7. LOGIC GAPS: Do not fill missing context with assumptions. If something is unclear, ask a clarifying question before answering.
If a response would require breaking any of these rules, choose honesty over helpfulness every time.

## What this repo is

Rebuild of the Siemens Healthineers 2026 hackathon (Challenge C2 "Teach the robot") work on the
FR3 Duo cube-stacking task, done to reproduce and honestly test the pitched result (pi0.5 LoRA:
0 % success after 5k steps on the organizer data, 54 % after +2.5k steps on 200 native MuJoCo
episodes). All code lives in `fr3duo_pipeline/`. `RESULTS.md` is the only place results are
reported; numbers there come from committed JSON under `results/`, never from memory.

## Facts that are easy to get wrong

- The training dataset is `fr3_duo_rand500` (490 episodes, **6 fps**, task "stack the three cubes
  into one tower", 16-D absolute joint actions, gripper action binary 1 = open). It is NOT the
  handover dataset, and it was rendered in MuJoCo (robot-control-stack FR3 Duo scene), not Isaac Sim,
  despite the organizer folder name. `info.json` `splits.train: "0:41"` is wrong; ignore it.
- Organizer zips in `C2_Teach-the-robot_VLA/` are git-ignored and must never be duplicated; unzip
  once with `fr3duo_pipeline/tools/unpack_dataset.sh` into `data/`.
- Two Python environments, never one:
  - MuJoCo / scene / expert / eval sim side: `/home/robolab-2/ur5e_hande_workspace/robot-control-stack/.venv`
    (rcs 0.7.2, mujoco 3.10, Python 3.11). Run with `MUJOCO_GL=egl`. Never edit that repo.
  - LeRobot 0.4.4 side (dataset writing, training smoke, policy server):
    `/home/robolab-2/ur5e_hande_workspace/Thesis/thesis/envs/pi0/bin/python`. lerobot 0.6.x is NOT
    used anywhere because the HPC venv is 0.4.4.
- Training runs on the Fraunhofer IIS HPC in the thesis venv (`~/thesis/scripts/hpc/.venv`, V100s,
  no bf16, fp16 autocast). This workstation cannot reach the HPC; the user pulls this repo there.
- The local RTX 5070 Ti (16 GB) is shared with thesis jobs; check `nvidia-smi` before any GPU work.
- Commit and push only when the user asks.
