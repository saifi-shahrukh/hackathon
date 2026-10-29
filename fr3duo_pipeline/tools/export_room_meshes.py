"""Export the organizers' hospital room (Hackathon_FR3Duo.gltf) as per-prop OBJ meshes for MuJoCo.

    /home/robolab-2/miniconda3/envs/robo/bin/python fr3duo_pipeline/tools/export_room_meshes.py \
        C2_Teach-the-robot_VLA/Environment/Environment_Hackathon_2026/Hackathon_FR3Duo.gltf fr3duo_pipeline/assets/room

Every prop under the glTF `World_1` node (except the robot subtree, which RCS provides) is written
in the ROBOT ROOT frame (z-up, x forward, +y = left arm side; origin = the FR3 Duo base prim), one
OBJ per (prop, material) so each gets its own colour/texture in the MJCF. Base-colour textures are
saved as PNG next to the OBJ. `manifest.json` lists files, rgba, texture, bbox and vertex counts;
`tools/build_room_mjcf.py` turns it into the scene XML.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

ROBOT_NODE = "FR3Duo_withV_Mkit__1_"
SKIP = {ROBOT_NODE, "DomeLight", "RectLight"}
# USD (Y-up world) -> robot root (Z-up): inverse of the robot root's world transform, read from the file.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gltf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-verts", type=int, default=120000, help="decimate props above this vertex count")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sc = trimesh.load(str(a.gltf), force="scene")
    g = sc.graph
    ch = g.transforms.children
    T_root, _ = g.get(ROBOT_NODE)
    Rinv = np.linalg.inv(T_root)
    world1 = "World_1"

    def mesh_nodes_below(n):
        stack, out = [n], []
        while stack:
            x = stack.pop()
            if x in g.nodes_geometry:
                out.append(x)
            stack.extend(ch.get(x, []))
        return out

    manifest = {"source": str(a.gltf), "frame": "robot_root_zup", "props": {}}
    for prop in sorted(ch[world1]):
        if prop in SKIP:
            continue
        nodes = mesh_nodes_below(prop)
        if not nodes:
            continue
        by_mat = defaultdict(list)
        for m in nodes:
            T, geom = g[m]
            mesh = sc.geometry[geom].copy()
            mesh.apply_transform(Rinv @ T)
            vis = mesh.visual
            mat = getattr(vis, "material", None)
            rgba, tex, key = (0.7, 0.7, 0.7, 1.0), None, "default"
            if mat is not None:
                col = getattr(mat, "baseColorFactor", None)
                if col is None:
                    col = getattr(mat, "diffuse", None)
                if col is not None:
                    c = np.asarray(col, dtype=float)
                    if c.max() > 1.0:
                        c = c / 255.0
                    rgba = tuple(float(x) for x in (list(c[:4]) + [1.0])[:4])
                img = getattr(mat, "baseColorTexture", None)
                if img is None:
                    img = getattr(mat, "image", None)
                key = getattr(mat, "name", None) or "default"
                if img is not None:
                    tex = img
            by_mat[key].append((mesh, rgba, tex))
        entries = []
        for i, (key, items) in enumerate(by_mat.items()):
            meshes = [it[0] for it in items]
            rgba, tex = items[0][1], items[0][2]
            merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            if len(merged.vertices) > a.max_verts:
                try:
                    merged = merged.simplify_quadric_decimation(face_count=int(len(merged.faces) * a.max_verts / len(merged.vertices)))
                except Exception:  # noqa: BLE001
                    pass
            name = f"{prop}__{i}".replace(" ", "_")
            obj = a.out / f"{name}.obj"
            has_uv = tex is not None and hasattr(merged.visual, "uv") and merged.visual.uv is not None and len(merged.visual.uv) == len(merged.vertices)
            tex_file = None
            if has_uv:
                tex_file = a.out / f"{name}.png"
                try:
                    tex.save(tex_file)
                except Exception:  # noqa: BLE001
                    tex_file = None
            # write OBJ by hand (vertices, optional uv, faces); MuJoCo ignores MTL
            v = merged.vertices
            f = merged.faces + 1
            with obj.open("w") as fh:
                fh.write(f"# {prop} material {key} in robot root frame\n")
                fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
                if tex_file is not None:
                    fh.writelines(f"vt {u:.6f} {w:.6f}\n" for u, w in merged.visual.uv)
                    fh.writelines(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n" for i0, i1, i2 in f)
                else:
                    fh.writelines(f"f {i0} {i1} {i2}\n" for i0, i1, i2 in f)
            lo, hi = v.min(0), v.max(0)
            entries.append({"file": obj.name, "material": key, "rgba": list(rgba), "texture": tex_file.name if tex_file else None,
                            "verts": int(len(v)), "faces": int(len(f)), "bbox_min": lo.round(4).tolist(), "bbox_max": hi.round(4).tolist()})
        allv = np.concatenate([np.array([e["bbox_min"], e["bbox_max"]]) for e in entries])
        manifest["props"][prop] = {"parts": entries, "bbox_min": allv.min(0).round(4).tolist(), "bbox_max": allv.max(0).round(4).tolist()}
        print(f"[export] {prop:45s} parts={len(entries)} verts={sum(e['verts'] for e in entries):7d} tex={sum(e['texture'] is not None for e in entries)} "
              f"bbox {allv.min(0).round(3)} {allv.max(0).round(3)}")
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[export] {len(manifest['props'])} props -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
