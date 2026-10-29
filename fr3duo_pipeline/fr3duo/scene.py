"""The cube-stacking scene on top of RCS's FR3 Duo MuJoCo env (RCS venv only).

What the organizer data shows (fr3duo_pipeline/configs/scene_calib.json, head-camera frames):
- three cubes of ~4 cm on a flat surface at world z = 0 (grasp TCP height 3.74 cm, spread 3 mm);
- spawn region x in [0.35, 0.65], y in [-0.15, 0.15] (world frame, +y is the left arm's side);
- the base cube A is one of the three at random; of the other two, the LEFT arm always picks the
  one with the larger y (B) and the RIGHT arm the one with the smaller y (C). B_y > C_y held in
  490/490 episodes, and the three observed y-orders are exactly those this rule allows;
- a wood-textured table surface with a random tint, a flat random-coloured mat beyond it, the
  RCS checker floor further out, random light intensity, random cube colours.
The organizers' generator itself is not available, so the look is an approximation; the
geometry (robot, cameras, cube size, spawn rule, control rate) is measured.

All randomisation happens at runtime on the compiled model (rgba, light, free-joint qpos), so one
env serves every episode.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .rcs_scene import DuoHandles, make_duo_env, obs_to_state16

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

CUBE_XML = """<mujoco model="{name}">
  <worldbody>
    <body name="{name}">
      <geom name="{name}_geom" type="box" size="{h} {h} {h}" rgba="{rgba}" friction="1 0.3 0.1"
            density="{density}" solref="0.004 1" solimp="0.95 0.99 0.001" group="1"/>
      <joint type="free" name="{name}_joint"/>
    </body>
  </worldbody>
</mujoco>
"""

TABLE_XML = """<mujoco model="table">
  <asset>
    <texture name="wood" type="2d" file="{wood_png}"/>
    <material name="wood" texture="wood" texrepeat="{repx} {repy}" texuniform="false" rgba="1 1 1 1" reflectance="0.05"/>
    <material name="mat" rgba="0.3 0.9 0.9 1" reflectance="0.0"/>
  </asset>
  <worldbody>
    <body name="table">
      <geom name="mat_geom" type="box" size="{mx} {my} 0.004" pos="{mx_off} 0 -0.0055" material="mat" contype="0" conaffinity="0" group="1"/>
      <geom name="table_geom" type="box" size="{tx} {ty} 0.005" pos="{tx_off} 0 -0.0049" material="wood" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def wood_texture_png(path: Path, size: int = 512, seed: int = 0) -> Path:
    """Procedural plank-like texture (grey-scale, tinted at runtime through the material rgba)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    # straight grain along the texture's y axis (mapped to world x: planks run away from the head camera)
    grain = np.zeros_like(x)
    for f, a in ((9, 0.45), (23, 0.25), (57, 0.15), (131, 0.08)):
        grain += a * np.sin(2 * np.pi * (f * x + 0.03 * np.sin(2 * np.pi * 2 * y + rng.uniform(0, 6.28))))
    grain += 0.06 * rng.standard_normal((size, size)).astype(np.float32)
    img = 0.74 + 0.14 * grain
    # two plank seams (one per half), running along the grain
    seams = (np.abs(((x * 2) % 1.0) - 0.5) > 0.492)
    img[seams] *= 0.6
    img = np.clip(img, 0, 1).T   # grain and seams along the world x axis (away from the head camera)
    rgb = (np.stack([img, img * 0.93, img * 0.85], -1) * 255).astype(np.uint8)
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


@dataclass
class SceneParams:
    cube_half: float = 0.021           # 4.2 cm cubes: gripper reading 0.521 when held (dataset 0.522)
    cube_density: float = 300.0
    table_half: tuple[float, float] = (0.60, 0.80)   # tuned visually against the head camera
    table_x_offset: float = 0.55
    mat_half: tuple[float, float] = (1.6, 1.6)
    mat_x_offset: float = 0.9
    spawn_x: tuple[float, float] = (0.35, 0.65)
    spawn_y: tuple[float, float] = (-0.15, 0.15)
    min_dist: float = 0.07
    cube_palette: list = field(default_factory=lambda: [
        (0.9, 0.1, 0.1), (0.1, 0.2, 0.9), (0.1, 0.8, 0.2), (0.95, 0.5, 0.1), (0.1, 0.8, 0.85),
        (0.6, 0.1, 0.7), (0.95, 0.9, 0.1), (0.9, 0.3, 0.6), (0.3, 0.3, 0.3), (0.0, 0.5, 0.5)])
    table_tints: list = field(default_factory=lambda: [
        (1.0, 0.65, 0.68), (1.0, 0.9, 0.78), (1.0, 0.8, 0.7), (0.95, 0.85, 0.65), (0.8, 0.35, 0.45),
        (0.9, 0.85, 0.75), (0.75, 0.7, 0.45), (1.0, 0.75, 0.8), (0.95, 0.7, 0.5), (0.85, 0.85, 0.85)])
    mat_colors: list = field(default_factory=lambda: [
        (0.5, 0.95, 0.95), (0.95, 0.95, 0.95), (0.2, 0.2, 0.9), (0.95, 0.9, 0.3), (0.8, 0.1, 0.2),
        (0.3, 0.6, 0.3), (0.5, 0.5, 0.5), (0.9, 0.6, 0.2), (0.6, 0.3, 0.1), (0.1, 0.1, 0.1)])
    light_scale: tuple[float, float] = (0.25, 1.4)


class CubeScene:
    """FR3 Duo + table + three cubes, 6 Hz absolute joint control, 224x224 cameras."""

    def __init__(self, params: SceneParams | None = None, *, cam_res: int = 224, headless: bool = True,
                 frequency: float = 6.0, asset_dir: Path | None = None, overview: bool = False):
        self.p = params or SceneParams()
        from rcs._core.common import Pose
        adir = Path(asset_dir) if asset_dir else Path(tempfile.mkdtemp(prefix="fr3duo_assets_"))
        adir.mkdir(parents=True, exist_ok=True)
        wood = wood_texture_png(adir / "wood.png")
        objs = {}
        table_xml = adir / "table.xml"
        table_xml.write_text(TABLE_XML.format(
            wood_png=str(wood), repx=3, repy=4, tx=self.p.table_half[0], ty=self.p.table_half[1], tx_off=0.0,
            mx=self.p.mat_half[0], my=self.p.mat_half[1], mx_off=0.0))
        objs["table"] = (str(table_xml), Pose(translation=np.array([self.p.table_x_offset, 0.0, 0.0]), quaternion=np.array([0, 0, 0, 1.0])))
        for n in ("cube_a", "cube_b", "cube_c"):
            xml = adir / f"{n}.xml"
            xml.write_text(CUBE_XML.format(name=n, h=self.p.cube_half, rgba="0.8 0.1 0.1 1", density=self.p.cube_density))
            objs[n] = (str(xml), Pose(translation=np.array([0.5, 0.0, 0.3]), quaternion=np.array([0, 0, 0, 1.0])))
        ov = ((1.3, -1.1, 0.95), (0.5, 0.0, 0.05)) if overview else None
        self.env, self.cfg = make_duo_env(headless=headless, frequency=frequency, cam_res=cam_res, world_frame_objects=objs, overview=ov)
        self.env.reset()
        self.h = DuoHandles.from_env(self.env)
        m = self.h.model
        self.cube_names = ["cube_a", "cube_b", "cube_c"]
        self.cube_bid = {n: self._find(mujoco.mjtObj.mjOBJ_BODY, n) for n in self.cube_names}
        self.cube_gid = {n: self._find(mujoco.mjtObj.mjOBJ_GEOM, n + "_geom") for n in self.cube_names}
        self.cube_jadr = {n: int(m.jnt_qposadr[self._find(mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")]) for n in self.cube_names}
        self.cube_dadr = {n: int(m.jnt_dofadr[self._find(mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")]) for n in self.cube_names}
        self.table_gid = self._find(mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        self.mat_gid = self._find(mujoco.mjtObj.mjOBJ_GEOM, "mat_geom")
        self.table_matid = int(m.geom_matid[self.table_gid])
        self.light0 = (m.light_diffuse.copy(), m.light_ambient.copy())
        self.last_spawn = None
        self.rng = np.random.default_rng(0)

    def _find(self, kind, suffix: str) -> int:
        m = self.h.model
        n = {mujoco.mjtObj.mjOBJ_BODY: m.nbody, mujoco.mjtObj.mjOBJ_GEOM: m.ngeom, mujoco.mjtObj.mjOBJ_JOINT: m.njnt}[kind]
        for i in range(n):
            name = mujoco.mj_id2name(m, kind, i)
            if name and name.endswith(suffix):
                return i
        raise KeyError(f"no {kind} ending with '{suffix}'")

    # ── spawning ────────────────────────────────────────────────────────────────────────────────
    def sample_spawn(self, rng: np.random.Generator) -> dict:
        p = self.p
        while True:
            pts = np.stack([rng.uniform(p.spawn_x[0], p.spawn_x[1], 3), rng.uniform(p.spawn_y[0], p.spawn_y[1], 3)], 1)
            d = [np.linalg.norm(pts[i] - pts[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
            if min(d) >= p.min_dist:
                break
        a = int(rng.integers(3))
        others = [i for i in range(3) if i != a]
        b, c = (others[0], others[1]) if pts[others[0], 1] > pts[others[1], 1] else (others[1], others[0])
        return {"cube_a": pts[a], "cube_b": pts[b], "cube_c": pts[c]}

    def place_cubes(self, xy: dict, yaw: dict | None = None) -> None:
        d, m = self.h.data, self.h.model
        for n in self.cube_names:
            adr = self.cube_jadr[n]
            yw = 0.0 if yaw is None else float(yaw.get(n, 0.0))
            d.qpos[adr: adr + 3] = [xy[n][0], xy[n][1], self.p.cube_half + 0.0005]
            d.qpos[adr + 3: adr + 7] = [np.cos(yw / 2), 0, 0, np.sin(yw / 2)]
            d.qvel[self.cube_dadr[n]: self.cube_dadr[n] + 6] = 0.0
        mujoco.mj_forward(m, d)

    def randomize_look(self, rng: np.random.Generator) -> dict:
        m, p = self.h.model, self.p
        cols = rng.choice(len(p.cube_palette), 3, replace=False)
        for n, ci in zip(self.cube_names, cols):
            m.geom_rgba[self.cube_gid[n]] = (*p.cube_palette[ci], 1.0)
        tint = p.table_tints[int(rng.integers(len(p.table_tints)))]
        m.mat_rgba[self.table_matid] = (*tint, 1.0)
        mat = p.mat_colors[int(rng.integers(len(p.mat_colors)))]
        m.geom_rgba[self.mat_gid] = (*mat, 1.0)
        ls = float(rng.uniform(*p.light_scale))
        m.light_diffuse[:] = self.light0[0] * ls
        m.light_ambient[:] = self.light0[1] * min(1.0, ls)
        return {"cube_colors": [list(map(float, p.cube_palette[ci])) for ci in cols], "table_tint": list(tint),
                "mat_color": list(mat), "light_scale": ls}

    def reset(self, seed: int | None = None, *, xy: dict | None = None, randomize_look: bool = True) -> tuple[dict, dict]:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        obs, info = self.env.reset()
        look = self.randomize_look(rng) if randomize_look else {}
        spawn = xy or self.sample_spawn(rng)
        self.place_cubes(spawn)
        self.h.sim.step(30)   # let the cubes settle on the table
        obs = self._fresh_obs()
        self.last_spawn = {k: list(map(float, v)) for k, v in spawn.items()}
        return obs, {"spawn": self.last_spawn, "look": look}

    def _fresh_obs(self) -> dict:
        # re-observe without stepping: send the current joints as the action with async stepping of 0?
        # RCS has no observe-only call; use a zero-duration step by repeating the current pose.
        s = self.state16()
        from .rcs_scene import action_dict
        obs, *_ = self.env.step(action_dict(s))
        return obs

    # ── stepping and state ──────────────────────────────────────────────────────────────────────
    def step(self, action16: np.ndarray) -> dict:
        from .rcs_scene import action_dict
        obs, *_ = self.env.step(action_dict(np.asarray(action16, dtype=np.float64)))
        return obs

    def state16(self) -> np.ndarray:
        out = np.zeros(16, dtype=np.float32)
        for s, sl, gi in (("left", slice(0, 7), 7), ("right", slice(8, 15), 15)):
            out[sl] = self.h.arm_joints(s)
            out[gi] = self.h.gripper_obs(s)
        return out

    @staticmethod
    def images(obs: dict) -> dict:
        f = obs["frames"]
        out = {"external_camera": np.asarray(f["head"]["rgb"]["data"]).astype(np.uint8),
               "left_wrist_camera": np.asarray(f["left_wrist"]["rgb"]["data"]).astype(np.uint8),
               "right_wrist_camera": np.asarray(f["right_wrist"]["rgb"]["data"]).astype(np.uint8)}
        if "overview" in f:
            out["overview"] = np.asarray(f["overview"]["rgb"]["data"]).astype(np.uint8)
        return out

    def cube_pos(self) -> dict:
        return {n: self.h.data.xpos[self.cube_bid[n]].copy() for n in self.cube_names}

    def cube_vel(self) -> dict:
        return {n: self.h.data.qvel[self.cube_dadr[n]: self.cube_dadr[n] + 3].copy() for n in self.cube_names}

    def tower_status(self, xy_tol: float = 0.015, z_tol: float = 0.012) -> dict:
        """Is there a 3-cube vertical tower (any order) resting on the table?"""
        pos = self.cube_pos()
        h2 = 2 * self.p.cube_half
        names = sorted(self.cube_names, key=lambda n: pos[n][2])
        base, mid, top = names
        on_table = abs(pos[base][2] - self.p.cube_half) < z_tol
        mid_ok = abs(pos[mid][2] - pos[base][2] - h2) < z_tol and np.linalg.norm(pos[mid][:2] - pos[base][:2]) < xy_tol
        top_ok = abs(pos[top][2] - pos[mid][2] - h2) < z_tol and np.linalg.norm(pos[top][:2] - pos[mid][:2]) < xy_tol
        vel = self.cube_vel()
        still = all(np.linalg.norm(v) < 0.02 for v in vel.values())
        tcp = {s: self.h.tcp_world(s)[0] for s in ("left", "right")}
        released = all(np.linalg.norm(tcp[s] - pos[top]) > 0.03 or self.h.gripper_obs(s) > 0.9 for s in tcp)
        return {"success": bool(on_table and mid_ok and top_ok and still and released), "order": names,
                "on_table": bool(on_table), "mid_ok": bool(mid_ok), "top_ok": bool(top_ok), "still": bool(still),
                "released": bool(released), "pos": {n: pos[n].round(4).tolist() for n in self.cube_names}}

    def close(self):
        self.env.close()
