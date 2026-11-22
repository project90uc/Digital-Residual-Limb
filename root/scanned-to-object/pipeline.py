"""Point cloud of a negative cast -> watertight positive limb model (STL, millimetres).

Steps (each one saves its output, so you can stop, fix a file in Blender or
CloudCompare, and resume with --from):

  1 load       read .ply/.xyz/.txt/.csv, convert to mm, filter by confidence
  2 isolate    remove the table plane, keep the biggest cluster (the cast)
  3 clean      remove outlier points
  4 normals    estimate normals and point them OUTWARD from the limb
  5 surface    Screened Poisson reconstruction (watertight)
  6 solid      cut off what Poisson invented past the cast opening, cap it flat, export STL

Example:
  python pipeline.py scan.ply -o out/ --min-confidence 2
  python pipeline.py out/04_normals.ply -o out/ --from 5      # resume after editing
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import load_points

STEPS = ["load", "isolate", "clean", "normals", "surface", "solid"]


# ---------------------------------------------------------------- helpers
def principal_axis(points, up=None):
    """Long axis of the limb. If the table normal is known, the cast stands on the
    table, so the axis is the table normal (more stable than PCA for short casts)."""
    c = points.mean(axis=0)
    if up is not None:
        a = np.asarray(up, float)
    else:
        _, _, vt = np.linalg.svd(points - c, full_matrices=False)
        a = vt[0]
    return c, a / np.linalg.norm(a)


def outward_directions(points, center, axis):
    """Direction from the limb's axis to each point. Outward normals must agree with it.
    Using the axis (not the centroid) keeps this right for long, thin limbs."""
    rel = points - center
    t = rel @ axis
    radial = rel - np.outer(t, axis)
    r = np.median(np.linalg.norm(radial, axis=1))
    lo, hi = t.min() + r, t.max() - r
    if lo > hi:  # short, wide object: use the middle
        lo = hi = (t.min() + t.max()) / 2
    q = center + np.outer(np.clip(t, lo, hi), axis)
    return points - q


# ---------------------------------------------------------------- steps
def step_load(args):
    pc = load_points(args.input, units=args.units, min_confidence=args.min_confidence)
    return pc, {"points": len(pc.points)}


def step_isolate(pc, args, meta):
    if args.no_table:
        return pc, {}
    plane, inliers = pc.segment_plane(args.table_dist, 3, 3000)
    n = np.asarray(plane[:3])
    pts = np.asarray(pc.points)
    h = pts @ n + plane[3]
    if np.median(h) < 0:  # make "up" point toward the object
        n, h, plane = -n, -h, -np.asarray(plane)
    keep = h > args.table_dist
    obj = pc.select_by_index(np.where(keep)[0])
    labels = np.asarray(obj.cluster_dbscan(eps=args.cluster_eps, min_points=10))
    if (labels >= 0).sum() == 0:
        raise RuntimeError("No cluster found above the table. Try a larger --cluster-eps.")
    biggest = np.bincount(labels[labels >= 0]).argmax()
    obj = obj.select_by_index(np.where(labels == biggest)[0])
    return obj, {"table_normal": n.tolist(), "table_offset": float(plane[3]),
                 "table_points_removed": len(inliers), "points": len(obj.points)}


def step_clean(pc, args, meta):
    if args.voxel > 0:
        pc = pc.voxel_down_sample(args.voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=args.outlier_std)
    return pc, {"points": len(pc.points)}


def step_normals(pc, args, meta):
    pts = np.asarray(pc.points)
    spacing = np.median(np.asarray(pc.compute_nearest_neighbor_distance()))
    radius = max(spacing * 6, args.resolution * 3)  # wider for noisy scans so normals average out noise
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
    center, axis = principal_axis(pts, meta.get("table_normal"))
    out = outward_directions(pts, center, axis)
    n = np.asarray(pc.normals)
    flip = np.einsum("ij,ij->i", n, out) < 0
    n[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(n)
    return pc, {"axis": axis.tolist(), "center": center.tolist(),
                "point_spacing_mm": float(spacing), "normals_flipped": int(flip.sum())}


def step_surface(pc, args, meta):
    if args.poissonrecon:
        pts = np.asarray(pc.points)
        depth = args.depth or int(np.ceil(np.log2((pts.max(0) - pts.min(0)).max() * 1.1 / args.resolution)))
        mesh = _poissonrecon_binary(pc, args, depth)
        info = {"depth": depth}
    elif args.depth:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=args.depth, scale=1.1)
        info = {"depth": args.depth}
    else:  # width = size of the finest cells in mm (same units as the points)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=0, width=args.resolution, scale=1.1, linear_fit=False)
        info = {"cell_mm": args.resolution}
    info["triangles"] = len(mesh.triangles)
    return mesh, info


def _poissonrecon_binary(pc, args, depth):
    tmp = Path(tempfile.mkdtemp())
    o3d.io.write_point_cloud(str(tmp / "in.ply"), pc, write_ascii=False)
    subprocess.run([args.poissonrecon, "--in", str(tmp / "in.ply"), "--out", str(tmp / "out.ply"),
                    "--depth", str(depth), "--pointWeight", str(args.point_weight)], check=True)
    mesh = o3d.io.read_triangle_mesh(str(tmp / "out.ply"))
    shutil.rmtree(tmp)
    return mesh


def step_solid(mesh, args, meta):
    """Poisson closes every gap, including the cast opening where there is no data.
    Cut that invented part off with a plane just inside the last real points, and cap it flat."""
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=True)
    tm = _largest_component(tm)
    info = {}
    pts_file = Path(args.out) / "04_normals.ply"
    if not args.no_cap and "axis" in meta and pts_file.exists():
        pts = np.asarray(o3d.io.read_point_cloud(str(pts_file)).points)
        axis, center = np.asarray(meta["axis"]), np.asarray(meta["center"])
        t = (pts - center) @ axis
        for sign, label in ((1, "top"), (-1, "bottom")):
            if args.cap in (label, "both") or (args.cap == "auto" and _end_is_open(pts, t, axis, center, sign)):
                end = t.max() if sign > 0 else t.min()
                origin = center + axis * (end - sign * args.cap_margin)
                tm = tm.slice_plane(origin, -sign * axis, cap=True)
                info["capped_" + label] = True
        tm = _largest_component(tm)
    if not tm.is_watertight:  # repair small holes / non-manifold edges left by noisy scans
        import pymeshfix
        fix = pymeshfix.MeshFix(tm.vertices, tm.faces)
        fix.repair()
        tm = trimesh.Trimesh(fix.points, fix.faces)
        info["repaired"] = True
    trimesh.repair.fix_normals(tm)
    handles = max(0, (2 - tm.euler_number) // 2)
    if handles:
        info["warning"] = (f"surface has {handles} tunnel(s), usually scan noise. "
                           f"Try a larger resolution / Detail value (now {args.resolution} mm).")
    info.update({"watertight": bool(tm.is_watertight), "tunnels": int(handles),
                 "volume_cm3": float(tm.volume / 1000) if tm.is_watertight else None,
                 "size_mm": np.round(tm.extents, 1).tolist()})
    return tm, info


def _largest_component(tm):
    parts = tm.split(only_watertight=False)
    return max(parts, key=lambda m: len(m.faces)) if len(parts) > 1 else tm


def _end_is_open(pts, t, axis, center, sign):
    """Closed end (like the distal end) has points near the axis. Open end has only a rim."""
    span = t.max() - t.min()
    slab = t > t.max() - 0.05 * span if sign > 0 else t < t.min() + 0.05 * span
    rel = pts[slab] - center
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    all_rel = pts - center
    r_med = np.median(np.linalg.norm(all_rel - np.outer(all_rel @ axis, axis), axis=1))
    return (radial < 0.5 * r_med).mean() < 0.05


# ---------------------------------------------------------------- driver
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--from", dest="start", type=int, default=1, help="step to start at (1-6)")
    p.add_argument("--units", default="m", choices=["m", "cm", "mm"], help="units of the input file")
    p.add_argument("--min-confidence", type=float, default=None, help="drop points below this confidence")
    p.add_argument("--no-table", action="store_true", help="skip table removal")
    p.add_argument("--table-dist", type=float, default=6.0, help="mm; plane thickness")
    p.add_argument("--cluster-eps", type=float, default=12.0, help="mm")
    p.add_argument("--voxel", type=float, default=0.0, help="mm; 0 keeps every point")
    p.add_argument("--outlier-std", type=float, default=2.0)
    p.add_argument("--resolution", type=float, default=2.0,
                   help="mm; Poisson cell size. Smaller = more detail but also fits noise")
    p.add_argument("--depth", type=int, default=None, help="override Poisson depth directly")
    p.add_argument("--point-weight", type=float, default=10.0, help="PoissonRecon binary only")
    p.add_argument("--poissonrecon", default=None, help="path to mkazhdan PoissonRecon binary")
    p.add_argument("--cap", default="auto", choices=["auto", "top", "bottom", "both"],
                   help="which open end to cut flat; auto finds the end with no data")
    p.add_argument("--no-cap", action="store_true")
    p.add_argument("--cap-margin", type=float, default=2.0, help="mm cut inside each open end")
    return p


def run(args, log=print):
    """Run the pipeline. Returns meta (dict). `log` receives one line per step."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if args.start > 1 and meta_path.exists() else {}

    obj = None
    if args.start > 1:
        obj = (o3d.io.read_triangle_mesh(args.input) if args.start == 6
               else o3d.io.read_point_cloud(args.input))
        if args.start == 6 and not obj.has_triangles():
            raise ValueError("--from 6 needs a mesh file")

    for i, name in enumerate(STEPS, start=1):
        if i < args.start:
            continue
        log(f"[{i}/6] {name}...")
        if name == "load":
            obj, info = step_load(args)
        else:
            obj, info = globals()["step_" + name](obj, args, meta)
        meta.update(info)
        meta_path.write_text(json.dumps(meta, indent=2))
        stem = out / f"{i:02d}_{name}"
        if name == "solid":
            obj.export(stem.with_suffix(".stl"))
            obj.export(stem.with_suffix(".ply"))
        elif name == "surface":
            o3d.io.write_triangle_mesh(str(stem) + ".ply", obj)
        else:
            o3d.io.write_point_cloud(str(stem) + ".ply", obj)
        log(f"[{i}] {name:8s} {json.dumps(info)}")
    return meta


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
