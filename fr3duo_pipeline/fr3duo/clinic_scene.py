"""Cube stacking in the organizers' hospital room (Hackathon_FR3Duo USD rebuilt in MuJoCo).

Geometry comes from the USD/glTF (tools/export_room_meshes.py, tools/build_room_mjcf.py):
the FR3 Duo stands on its table, the serving tray sits on the supply cart in front-right of the
robot (tray interior ~307 x 229 mm, floor 33 mm below the robot base plane), the second cart to
the left, cabinets, shelves, bed, walls. The three cubes are spawned inside the serving tray and
stacked there. Cameras follow the USD: head camera on the head link (90 deg vertical FOV), D405
wrist cameras 20 deg off the tool axis (87 deg FOV); the robot kit itself is the same as RCS's.

Everything that CubeScene provides (stepping, state16, images, tower_status, look randomisation)
is reused; only the environment composition, spawn region and heights differ.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field  # noqa: F401
from pathlib import Path

import mujoco
import numpy as np

from .rcs_scene import DuoHandles, add_overview_camera, gl_environment_defaults, quiet_rcs_warnings
from .scene import CUBE_XML, CubeScene, SceneParams

ROOM = Path(__file__).resolve().parents[1] / "assets" / "room"


@dataclass
class ClinicParams(SceneParams):
    scene_xml: str = str(ROOM / "clinic_scene.xml")
    wrist_cam_xml: str = str(ROOM / "d405_fov87.xml")
    head_fovy: float = 90.0
    # spawn region: derived from the tray interior in scene_meta.json (root frame) minus this margin
    spawn_margin: float = 0.045
    spawn_x: tuple[float, float] = (0.0, 0.0)
    spawn_y: tuple[float, float] = (0.0, 0.0)
    min_dist: float = 0.065
    light_scale: tuple[float, float] = (0.6, 1.4)
    # "rand500": the cameras fr3_duo_rand500 was rendered with (RCS defaults: head camera on the mount,
    # fovy 52; D405 wrists fovy 58) so the policy trained on rand500 sees the same geometry.
    # "usd": the Isaac handover-stage cameras (head on the head link, fovy 90; wrists 87).
    cameras: str = "rand500"
    # cube colours as seen in rand500 (red, blue, green, orange, cyan, yellow, purple, pink)
    cube_palette: list = field(default_factory=lambda: [
        (0.9, 0.1, 0.1), (0.1, 0.2, 0.9), (0.1, 0.8, 0.2), (0.95, 0.5, 0.1), (0.1, 0.8, 0.85),
        (0.95, 0.9, 0.1), (0.6, 0.1, 0.7), (0.9, 0.3, 0.6)])


class ClinicScene(CubeScene):
    def __init__(self, params: ClinicParams | None = None, *, cam_res: int = 224, headless: bool = True,
                 frequency: float = 6.0, asset_dir: Path | None = None, overview: bool = False):
        self.p = params or ClinicParams()
        meta = json.loads((ROOM / "scene_meta.json").read_text())
        self.root_in_world = np.array(meta["root_in_world"])
        self.tray = meta["tray"]
        self.table_z_world = float(self.tray["floor_top"] + self.root_in_world[2])
        t, w, mg = self.tray, self.tray["wall"], self.p.spawn_margin
        if "spawn_root" in meta:   # simple layout: the rand500 cube region, for joint-space transfer
            self.p.spawn_x = tuple(meta["spawn_root"]["x"])
            self.p.spawn_y = tuple(meta["spawn_root"]["y"])
        else:
            self.p.spawn_x = (t["xmin"] + w + mg, t["xmax"] - w - mg)
            self.p.spawn_y = (t["ymin"] + w + mg, t["ymax"] - w - mg)
        from rcs._core.common import Pose
        from rcs.envs.scenes import CameraAdderConfig
        gl_environment_defaults()
        quiet_rcs_warnings()
        from rcs.envs.base import ControlMode, RelativeTo
        from rcs.envs.configs import EmptyWorldFR3Duo

        adir = Path(asset_dir) if asset_dir else Path(tempfile.mkdtemp(prefix="fr3duo_clinic_"))
        adir.mkdir(parents=True, exist_ok=True)
        objs = {}
        for n in ("cube_a", "cube_b", "cube_c"):
            xml = adir / f"{n}.xml"
            xml.write_text(CUBE_XML.format(name=n, h=self.p.cube_half, rgba="0.8 0.1 0.1 1", density=self.p.cube_density))
            objs[n] = (str(xml), Pose(translation=np.array([0.6, -0.5, self.table_z_world + 0.3]), quaternion=np.array([0, 0, 0, 1.0])))

        scene = EmptyWorldFR3Duo()
        cfg = scene.config()
        cfg.scene = self.p.scene_xml
        cfg.headless = headless
        cfg.control_mode = ControlMode.JOINTS
        cfg.relative_to = RelativeTo.NONE
        cfg.max_relative_movement = None
        cfg.sim_cfg.realtime = False
        cfg.sim_cfg.async_control = True
        cfg.sim_cfg.frequency = frequency
        for c in cfg.camera_cfgs.values():
            c.resolution_width = cam_res
            c.resolution_height = cam_res
        cfg.world_frame_objects = {**(cfg.world_frame_objects or {}), **objs}
        # cameras: default = RCS's own (what rand500 used); "usd" = the Isaac handover-stage cameras
        adds = dict(cfg.camera_adds)
        if self.p.cameras == "usd":
            for side in ("left", "right"):
                old = adds[f"{side}_wrist"]
                adds[f"{side}_wrist"] = CameraAdderConfig(xml_path=self.p.wrist_cam_xml, offset=old.offset, robot_name=side)
            hc = meta["head_camera_root"]
            pos = np.array(hc["pos"]) + self.root_in_world
            w, x, y, z = hc["quat_wxyz"]
            adds["head"] = CameraAdderConfig(xml_path=None, offset=Pose(translation=pos, quaternion=np.array([x, y, z, w])),
                                             fovy=self.p.head_fovy)
        cfg.camera_adds = adds
        if overview:
            tz = self.table_z_world
            add_overview_camera(cfg, (1.25, -1.05, tz + 0.75), (0.45, 0.0, tz + 0.05))
        self.env = scene.create_env(cfg)
        self.cfg = cfg
        self.env.reset()
        self.h = DuoHandles.from_env(self.env)
        m = self.h.model
        self.cube_names = ["cube_a", "cube_b", "cube_c"]
        self.cube_bid = {n: self._find(mujoco.mjtObj.mjOBJ_BODY, n) for n in self.cube_names}
        self.cube_gid = {n: self._find(mujoco.mjtObj.mjOBJ_GEOM, n + "_geom") for n in self.cube_names}
        self.cube_jadr = {n: int(m.jnt_qposadr[self._find(mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")]) for n in self.cube_names}
        self.cube_dadr = {n: int(m.jnt_dofadr[self._find(mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")]) for n in self.cube_names}
        self.light0 = (m.light_diffuse.copy(), m.light_ambient.copy())
        self.last_spawn = None
        self.rng = np.random.default_rng(0)

    # spawn region is given in the ROOT frame; convert to world
    def sample_spawn(self, rng: np.random.Generator) -> dict:
        p = self.p
        while True:
            pts = np.stack([rng.uniform(p.spawn_x[0], p.spawn_x[1], 3), rng.uniform(p.spawn_y[0], p.spawn_y[1], 3)], 1)
            d = [np.linalg.norm(pts[i] - pts[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
            if min(d) >= p.min_dist:
                break
        pts = pts + self.root_in_world[:2]
        a = int(rng.integers(3))
        others = [i for i in range(3) if i != a]
        b, c = (others[0], others[1]) if pts[others[0], 1] > pts[others[1], 1] else (others[1], others[0])
        return {"cube_a": pts[a], "cube_b": pts[b], "cube_c": pts[c]}

    def place_cubes(self, xy: dict, yaw: dict | None = None) -> None:
        d, m = self.h.data, self.h.model
        for n in self.cube_names:
            adr = self.cube_jadr[n]
            yw = 0.0 if yaw is None else float(yaw.get(n, 0.0))
            d.qpos[adr: adr + 3] = [xy[n][0], xy[n][1], self.table_z_world + self.p.cube_half + 0.0005]
            d.qpos[adr + 3: adr + 7] = [np.cos(yw / 2), 0, 0, np.sin(yw / 2)]
            d.qvel[self.cube_dadr[n]: self.cube_dadr[n] + 6] = 0.0
        mujoco.mj_forward(m, d)

    def randomize_look(self, rng: np.random.Generator) -> dict:
        m, p = self.h.model, self.p
        cols = rng.choice(len(p.cube_palette), 3, replace=False)
        for n, ci in zip(self.cube_names, cols):
            m.geom_rgba[self.cube_gid[n]] = (*p.cube_palette[ci], 1.0)
        ls = float(rng.uniform(*p.light_scale))
        m.light_diffuse[:] = self.light0[0] * ls
        m.light_ambient[:] = self.light0[1] * min(1.2, ls)
        return {"cube_colors": [list(map(float, p.cube_palette[ci])) for ci in cols], "light_scale": ls}

    def tower_status(self, xy_tol: float = 0.015, z_tol: float = 0.012) -> dict:
        pos = self.cube_pos()
        h2 = 2 * self.p.cube_half
        names = sorted(self.cube_names, key=lambda n: pos[n][2])
        base, mid, top = names
        on_table = abs(pos[base][2] - (self.table_z_world + self.p.cube_half)) < z_tol
        mid_ok = abs(pos[mid][2] - pos[base][2] - h2) < z_tol and np.linalg.norm(pos[mid][:2] - pos[base][:2]) < xy_tol
        top_ok = abs(pos[top][2] - pos[mid][2] - h2) < z_tol and np.linalg.norm(pos[top][:2] - pos[mid][:2]) < xy_tol
        still = all(np.linalg.norm(v) < 0.02 for v in self.cube_vel().values())
        tcp = {s: self.h.tcp_world(s)[0] for s in ("left", "right")}
        released = all(np.linalg.norm(tcp[s] - pos[top]) > 0.03 or self.h.gripper_obs(s) > 0.9 for s in tcp)
        return {"success": bool(on_table and mid_ok and top_ok and still and released), "order": names,
                "on_table": bool(on_table), "mid_ok": bool(mid_ok), "top_ok": bool(top_ok), "still": bool(still),
                "released": bool(released), "pos": {n: pos[n].round(4).tolist() for n in self.cube_names}}
