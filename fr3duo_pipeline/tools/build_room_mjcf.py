"""Build the MuJoCo base scene of the organizers' hospital room from the exported meshes.

    python fr3duo_pipeline/tools/build_room_mjcf.py            # -> fr3duo_pipeline/assets/room/clinic_scene.xml

Frames: the OBJ files are in the ROBOT ROOT frame of the USD (origin at the FR3 Duo base prim,
z-up). RCS composes the robot with link0 at world (0, +-0.0502, 0.342) while the USD has link0 at
root (0.0369, +-0.0502, 0.0509) with the same orientation, so the room is placed at
world offset ROOT_IN_WORLD = (-0.0369, 0, 0.2911). Everything else (cameras, spawn region) uses the
same offset (see fr3duo/clinic_scene.py).

Visual geoms carry the glTF base colours and textures. Collision geoms are explicit boxes for the
surfaces that matter (floor, robot table, both supply carts, the serving tray floor and its four
walls) measured from the mesh bounding boxes, mirroring the organizers' "static floor and four
static walls beneath the visible tray".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "assets" / "room"
ROOT_IN_WORLD = (-0.0369, 0.0, 0.2911)
USD_FLOOR_Z = -1.021
# serving tray (root frame, raw USD): bbox x 0.445-0.797, y -0.630..-0.372, z -0.039..0.008; interior 307 x 229 mm
TRAY_RAW = {"xmin": 0.445, "xmax": 0.797, "ymin": -0.630, "ymax": -0.372, "zmin": -0.039, "ztop": 0.008, "wall": 0.0225}
TRAY_FLOOR_TOP = -0.031   # overwritten by the mesh measurement below when available
# Stacking stage layout. The shipped USD is the HANDOVER stage (tray on the right cart, box on the
# left cart). At that position the tray is out of reach for a top-down grasp (IK error 15-23 cm,
# measured). The organizers' stacking overrides are not shipped, so the tray cart is moved in front
# of the robot: tray interior centred at root (0.463, 0.0), which is the centre of the cube
# workspace measured in fr3_duo_rand500 (x 0.35-0.65 world, y -0.15..0.15). The handover-only props
# (left cart, cardboard box, bottle, pill bottle) are removed from the stacking stage.
TRAY_SHIFT = (0.463 - (TRAY_RAW["xmin"] + TRAY_RAW["xmax"]) / 2, 0.0 - (TRAY_RAW["ymin"] + TRAY_RAW["ymax"]) / 2)
PROP_OFFSETS = {"SM_SupplyCart_01e": TRAY_SHIFT, "ServingTray": TRAY_SHIFT}
DROP = {"SM_SupplyCart_03a", "WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01", "NaturalBostonRoundBottle_A02_PR_NVD_01", "SM_PillBottle_01v"}
# Minimal stacking setup (default): only what the task needs and what the cameras see around it.
# The robot table ("Danny"), the supply cart with the serving tray, the floor and the four walls as
# background. Everything else in the room (cabinets, shelves, bed, desk, props) is left out: it costs
# render time, adds nothing to the task, and the policy cameras barely see it.
KEEP_MINIMAL = {"Danny", "SM_SupplyCart_01e", "ServingTray", "Floor", "Wall", "Wall1", "Wall2", "Wall3"}
TRAY = {**TRAY_RAW, "xmin": TRAY_RAW["xmin"] + TRAY_SHIFT[0], "xmax": TRAY_RAW["xmax"] + TRAY_SHIFT[0],
        "ymin": TRAY_RAW["ymin"] + TRAY_SHIFT[1], "ymax": TRAY_RAW["ymax"] + TRAY_SHIFT[1]}


def tray_floor_from_mesh() -> float | None:
    obj = ROOT / "ServingTray__0.obj"
    if not obj.exists():
        return None
    v = np.array([[float(t) for t in line.split()[1:4]] for line in obj.read_text().splitlines() if line.startswith("v ")])
    T = TRAY_RAW
    inside = (v[:, 0] > T["xmin"] + 0.05) & (v[:, 0] < T["xmax"] - 0.05) & (v[:, 1] > T["ymin"] + 0.05) & (v[:, 1] < T["ymax"] - 0.05)
    if inside.sum() < 4:
        return None
    return float(np.percentile(v[inside, 2], 95))   # top surface of the interior floor


# Simple layout (default): the organizers' stacking stage as seen in their head-camera frames, a plain
# raised tray-table with a thin chrome rim in front of the robot. Geometry chosen so the policy
# trained on fr3_duo_rand500 transfers: the table top sits at the rand500 grasp plane (world z = 0,
# 29 cm below the robot base plane) and the cubes spawn in the rand500 region (world x 0.35-0.65,
# y -0.15..0.15). Dimensions below are in WORLD metres (RCS frame: robot bases at z 0.342).
SIMPLE = {"top_z": 0.0, "x": (0.22, 0.78), "y": (-0.33, 0.33), "top_thickness": 0.02, "rim_radius": 0.008,
          "leg_radius": 0.012, "spawn_x": (0.35, 0.65), "spawn_y": (-0.15, 0.15)}


def build_simple(floor_top_unused: float) -> int:
    ox, oy, oz = ROOT_IN_WORLD
    S = SIMPLE
    x0, x1 = S["x"]; y0, y1 = S["y"]; z = S["top_z"]; th = S["top_thickness"]; rr = S["rim_radius"]
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    floor_z = USD_FLOOR_Z + oz
    man = json.loads((ROOT / "manifest.json").read_text())
    danny = man["props"]["Danny"]["parts"][0]
    floor = man["props"]["Floor"]["parts"][0]
    legs = "\n".join(
        f'      <geom name="leg{i}" type="cylinder" size="{S["leg_radius"]} {(z - th - floor_z) / 2:.4f}" pos="{lx:.3f} {ly:.3f} {(z - th + floor_z) / 2:.4f}" material="chrome" class="vis"/>'
        for i, (lx, ly) in enumerate(((x0 + 0.04, y0 + 0.04), (x0 + 0.04, y1 - 0.04), (x1 - 0.04, y0 + 0.04), (x1 - 0.04, y1 - 0.04))))
    rims = "\n".join([
        f'      <geom name="rim_x0" type="cylinder" size="{rr} {hy:.4f}" pos="{x0:.3f} {cy:.3f} {z + rr:.4f}" quat="0.7071 0.7071 0 0" material="chrome" class="vis"/>',
        f'      <geom name="rim_x1" type="cylinder" size="{rr} {hy:.4f}" pos="{x1:.3f} {cy:.3f} {z + rr:.4f}" quat="0.7071 0.7071 0 0" material="chrome" class="vis"/>',
        f'      <geom name="rim_y0" type="cylinder" size="{rr} {hx:.4f}" pos="{cx:.3f} {y0:.3f} {z + rr:.4f}" quat="0.7071 0 0.7071 0" material="chrome" class="vis"/>',
        f'      <geom name="rim_y1" type="cylinder" size="{rr} {hx:.4f}" pos="{cx:.3f} {y1:.3f} {z + rr:.4f}" quat="0.7071 0 0.7071 0" material="chrome" class="vis"/>'])
    xml = f"""<mujoco model="fr3 scene">
  <!-- generated by fr3duo_pipeline/tools/build_room_mjcf.py (simple layout) -->
  <compiler meshdir="." texturedir="." angle="radian"/>
  <option integrator="implicitfast" impratio="20" noslip_iterations="5" cone="elliptic">
    <flag multiccd="enable"/>
  </option>
  <statistic center="0.5 0 0.3" extent="1.5"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.7 0.7 0.72 1"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <map znear="0.01" zfar="30"/>
  </visual>
  <default>
    <default class="vis"><geom contype="0" conaffinity="0" group="1"/></default>
    <default class="col"><geom contype="1" conaffinity="1" group="3" rgba="0.8 0.2 0.2 0"/></default>
  </default>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.72 0.73 0.75" rgb2="0.72 0.73 0.75" width="64" height="64"/>
    <material name="mat_danny" rgba="0.92 0.92 0.93 1" specular="0.1" shininess="0.2"/>
    <mesh name="floor_mesh" file="{floor['file']}" inertia="shell"/>
    <texture name="tex_floor" type="2d" file="{floor['texture']}"/>
    <material name="mat_floor" texture="tex_floor" rgba="1 1 1 1" specular="0.02" shininess="0.05"/>
    <material name="tabletop" rgba="0.86 0.87 0.88 1" specular="0.3" shininess="0.4" reflectance="0.05"/>
    <material name="chrome" rgba="0.75 0.76 0.78 1" specular="0.8" shininess="0.8" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="key" pos="{1.2 + ox:.3f} {-1.6 + oy:.3f} {2.2 + oz:.3f}" dir="-0.35 0.5 -0.8" directional="false" diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15" cutoff="80" castshadow="true"/>
    <light name="fill" pos="{-1.0 + ox:.3f} {1.5 + oy:.3f} {2.0 + oz:.3f}" dir="0.4 -0.3 -0.86" directional="true" diffuse="0.25 0.25 0.25" specular="0 0 0" castshadow="false"/>
    <body name="room" pos="{ox} {oy} {oz}">
      <geom name="vis_danny" type="box" size="0.312 0.292 0.508" pos="-0.2 -0.034 -0.508" material="mat_danny" class="vis"/>
      <geom name="vis_floor" type="mesh" mesh="floor_mesh" material="mat_floor" class="vis"/>
      <geom name="col_floor" type="plane" size="8 8 0.1" pos="1 -1.5 {USD_FLOOR_Z}" class="col"/>
      <geom name="col_robot_table" type="box" size="0.312 0.292 0.02" pos="-0.2 -0.034 -0.02" class="col"/>
    </body>
    <body name="table">
      <geom name="tabletop" type="box" size="{hx:.4f} {hy:.4f} {th / 2:.4f}" pos="{cx:.3f} {cy:.3f} {z - th / 2:.4f}" material="tabletop" class="vis"/>
      <geom name="col_tabletop" type="box" size="{hx:.4f} {hy:.4f} {th / 2:.4f}" pos="{cx:.3f} {cy:.3f} {z - th / 2:.4f}" class="col" friction="1 0.3 0.1"/>
{rims}
{legs}
    </body>
  </worldbody>
</mujoco>
"""
    out = ROOT / "clinic_scene.xml"
    out.write_text(xml)
    # scene_meta in the ROOT frame (what ClinicScene reads): tray = the table top, spawn = rand500 region
    tray = {"xmin": x0 - ox, "xmax": x1 - ox, "ymin": y0 - oy, "ymax": y1 - oy, "zmin": z - th - oz, "ztop": z + 2 * rr - oz,
            "wall": 2 * rr, "floor_top": z - oz}
    meta = {"layout": "simple", "root_in_world": ROOT_IN_WORLD, "tray": tray, "usd_floor_z": USD_FLOOR_Z,
            "spawn_root": {"x": [S["spawn_x"][0] - ox, S["spawn_x"][1] - ox], "y": [S["spawn_y"][0] - oy, S["spawn_y"][1] - oy]},
            "table_world": S, "head_camera_root": {"pos": [0.0991, -0.0431, 0.4706], "quat_wxyz": [0.6613, 0.2930, -0.2942, -0.6248], "fovy": 90.0},
            "wrist_camera_fovy": 87.0}
    (ROOT / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[room] layout=simple: {out}; table top world z={z}, spawn world x={S['spawn_x']} y={S['spawn_y']}")
    return 0


def main() -> int:
    if "--layout" in sys.argv and sys.argv[sys.argv.index("--layout") + 1] == "simple" or "--layout" not in sys.argv and "--full-room" not in sys.argv and "--minimal" not in sys.argv:
        return build_simple(0.0)
    man = json.loads((ROOT / "manifest.json").read_text())
    floor_top = tray_floor_from_mesh()
    if floor_top is None:
        floor_top = TRAY_FLOOR_TOP
    print(f"[room] tray interior floor top (root frame) z = {floor_top:.4f}")
    assets, geoms = [], []
    minimal = "--full-room" not in sys.argv
    for prop, info in man["props"].items():
        if prop in DROP or (minimal and prop not in KEEP_MINIMAL):
            continue
        off = PROP_OFFSETS.get(prop, (0.0, 0.0))
        pos_attr = f' pos="{off[0]:.4f} {off[1]:.4f} 0"' if off != (0.0, 0.0) else ""
        for part in info["parts"]:
            mesh_name = part["file"][:-4]
            assets.append(f'    <mesh name="{mesh_name}" file="{part["file"]}" inertia="shell"/>')   # flat meshes (floor, walls) have no volume
            r, g, b, _a = part["rgba"]
            if part["texture"]:
                assets.append(f'    <texture name="tex_{mesh_name}" type="2d" file="{part["texture"]}"/>')
                assets.append(f'    <material name="mat_{mesh_name}" texture="tex_{mesh_name}" rgba="1 1 1 1" specular="0.05" shininess="0.1"/>')
            else:
                assets.append(f'    <material name="mat_{mesh_name}" rgba="{r:.3f} {g:.3f} {b:.3f} 1" specular="0.1" shininess="0.2"/>')
            geoms.append(f'      <geom name="vis_{mesh_name}" type="mesh" mesh="{mesh_name}" material="mat_{mesh_name}"{pos_attr} class="roomvis"/>')
    T = TRAY
    w = T["wall"]
    cx, cy = (T["xmin"] + T["xmax"]) / 2, (T["ymin"] + T["ymax"]) / 2
    hx, hy = (T["xmax"] - T["xmin"]) / 2, (T["ymax"] - T["ymin"]) / 2
    wall_h = (T["ztop"] - floor_top) / 2
    wall_z = floor_top + wall_h
    sx, sy = TRAY_SHIFT
    colliders = [
        f'      <geom name="col_floor" type="plane" size="8 8 0.1" pos="1 -1.5 {USD_FLOOR_Z}" class="roomcol"/>',
        # robot table "Danny": top at z=0, bbox x -0.512..0.112, y -0.326..0.258
        '      <geom name="col_robot_table" type="box" size="0.312 0.292 0.02" pos="-0.2 -0.034 -0.02" class="roomcol"/>',
        # supply cart 01e (tray cart, shifted with the tray): shelf under the tray; raw bbox x 0.365..0.907, y -0.985..0.122
        f'      <geom name="col_cart01_shelf" type="box" size="0.271 0.553 0.01" pos="{0.636 + sx:.4f} {-0.431 + sy:.4f} {T["zmin"] - 0.01:.4f}" class="roomcol"/>',
        # serving tray: floor + 4 walls
        f'      <geom name="col_tray_floor" type="box" size="{hx:.4f} {hy:.4f} 0.004" pos="{cx:.4f} {cy:.4f} {floor_top - 0.004:.4f}" class="roomcol" friction="1 0.3 0.1"/>',
        f'      <geom name="col_tray_wall_xmin" type="box" size="{w / 2:.4f} {hy:.4f} {wall_h:.4f}" pos="{T["xmin"] + w / 2:.4f} {cy:.4f} {wall_z:.4f}" class="roomcol"/>',
        f'      <geom name="col_tray_wall_xmax" type="box" size="{w / 2:.4f} {hy:.4f} {wall_h:.4f}" pos="{T["xmax"] - w / 2:.4f} {cy:.4f} {wall_z:.4f}" class="roomcol"/>',
        f'      <geom name="col_tray_wall_ymin" type="box" size="{hx:.4f} {w / 2:.4f} {wall_h:.4f}" pos="{cx:.4f} {T["ymin"] + w / 2:.4f} {wall_z:.4f}" class="roomcol"/>',
        f'      <geom name="col_tray_wall_ymax" type="box" size="{hx:.4f} {w / 2:.4f} {wall_h:.4f}" pos="{cx:.4f} {T["ymax"] - w / 2:.4f} {wall_z:.4f}" class="roomcol"/>',
    ]
    ox, oy, oz = ROOT_IN_WORLD
    xml = f"""<mujoco model="fr3 scene">
  <!-- generated by fr3duo_pipeline/tools/build_room_mjcf.py from the organizers' Hackathon_FR3Duo.gltf -->
  <compiler meshdir="." texturedir="." angle="radian"/>
  <option integrator="implicitfast" impratio="20" noslip_iterations="5" cone="elliptic">
    <flag multiccd="enable"/>
  </option>
  <statistic center="0.5 0 0.5" extent="2"/>
  <visual>
    <headlight diffuse="0.25 0.25 0.25" ambient="0.22 0.22 0.22" specular="0.05 0.05 0.05"/>
    <rgba haze="0.6 0.65 0.7 1"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <map znear="0.01" zfar="40"/>
  </visual>
  <default>
    <default class="roomvis"><geom contype="0" conaffinity="0" group="1"/></default>
    <default class="roomcol"><geom contype="1" conaffinity="1" group="3" rgba="0.8 0.2 0.2 0.0"/></default>
  </default>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.75 0.8 0.85" rgb2="0.75 0.8 0.85" width="64" height="64"/>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <!-- the organizers' rect light (root frame (1.86, -2.47, 2.10)) plus a soft fill -->
    <light name="rect_light" pos="{1.86 + ox:.3f} {-2.47 + oy:.3f} {2.10 + oz:.3f}" dir="-0.3 0.45 -0.84" directional="false" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1" cutoff="80" exponent="1" castshadow="true"/>
    <light name="fill_light" pos="{-1.0 + ox:.3f} {1.5 + oy:.3f} {2.0 + oz:.3f}" dir="0.4 -0.3 -0.86" directional="true" diffuse="0.2 0.2 0.2" specular="0 0 0" castshadow="false"/>
    <body name="room" pos="{ox} {oy} {oz}">
{chr(10).join(geoms)}
{chr(10).join(colliders)}
    </body>
  </worldbody>
</mujoco>
"""
    out = ROOT / "clinic_scene.xml"
    out.write_text(xml)
    meta = {"root_in_world": ROOT_IN_WORLD, "tray": {**T, "floor_top": floor_top}, "tray_raw": TRAY_RAW, "tray_shift": TRAY_SHIFT,
            "dropped_props": sorted(DROP), "usd_floor_z": USD_FLOOR_Z,
            "head_camera_root": {"pos": [0.0991, -0.0431, 0.4706], "quat_wxyz": [0.6613, 0.2930, -0.2942, -0.6248], "fovy": 90.0},
            "wrist_camera_fovy": 87.0}
    (ROOT / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    meta["layout"] = "minimal" if minimal else "full_room"
    (ROOT / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[room] layout={'minimal' if minimal else 'full room'}: {out} ({len(geoms)} visual geoms, {len(colliders)} colliders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
