"""Accuracy check: compare a scan or a reconstructed mesh against the true model.

  python compare.py out/06_solid.stl reference_limb.stl
  python compare.py out/03_clean.ply reference_limb.stl --test-units mm

Aligns the two (PCA, then ICP), measures every test point's distance to the reference
surface, fits a uniform scale factor, and writes:
  <out>/report.json      mean / median / p95 / max error in mm, scale error
  <out>/error_report.png histogram, error along the limb, cross-section overlay
  <out>/error_colored.ply test points colored by error (open in CloudCompare/MeshLab)
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize_scalar

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import load_points, UNIT_TO_MM


def load_test(path, units):
    path = Path(path)
    if path.suffix.lower() in (".stl", ".obj", ".off", ".glb") or _ply_has_faces(path):
        tm = trimesh.load(path, force="mesh")
        pts, _ = trimesh.sample.sample_surface_even(tm, 60000, seed=0)
        return np.asarray(pts) * UNIT_TO_MM[units]
    return np.asarray(load_points(path, units=units).points)


def _ply_has_faces(path):
    if path.suffix.lower() != ".ply":
        return False
    with open(path, "rb") as f:
        head = f.read(2000).decode("ascii", errors="replace")
    return "element face" in head and "element face 0" not in head


def pca_frame(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt


def align(test, ref_pc, ref_pts):
    """Try the PCA axis sign combinations, refine each with ICP, keep the best."""
    ct, at = pca_frame(test)
    cr, ar = pca_frame(ref_pts)
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(test))
    scale = np.linalg.norm(ref_pts.max(0) - ref_pts.min(0))
    best = None
    for signs in itertools.product([1, -1], repeat=2):
        s = np.array([signs[0], signs[1], signs[0] * signs[1]])  # keep it a rotation
        R = ar.T @ np.diag(s) @ at
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = cr - R @ ct
        for dist in (scale * 0.1, scale * 0.02):
            res = o3d.pipelines.registration.registration_icp(
                src, ref_pc, dist, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
            T = res.transformation
        if best is None or (res.fitness, -res.inlier_rmse) > (best[1].fitness, -best[1].inlier_rmse):
            best = (T, res)
    T = best[0]
    return test @ T[:3, :3].T + T[:3, 3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", help="scan points or reconstructed mesh")
    p.add_argument("reference", help="true model (mesh), e.g. the STL the cast was printed from")
    p.add_argument("--test-units", default="mm", choices=["m", "cm", "mm"])
    p.add_argument("--ref-units", default="mm", choices=["m", "cm", "mm"])
    p.add_argument("--no-align", action="store_true", help="files already share a coordinate frame")
    p.add_argument("--no-crop", action="store_true", help="score every test point, even past the reference's ends")
    p.add_argument("--crop-margin", type=float, default=5.0, help="mm trimmed off each end before scoring")
    p.add_argument("-o", "--out", default="compare_out")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = trimesh.load(args.reference, force="mesh")
    ref.apply_scale(UNIT_TO_MM[args.ref_units])
    ref_pts, fi = trimesh.sample.sample_surface_even(ref, 80000, seed=1)
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_pts))
    ref_pc.normals = o3d.utility.Vector3dVector(ref.face_normals[fi])

    test = load_test(args.test, args.test_units)
    if not args.no_align:
        test = align(test, ref_pc, ref_pts)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(ref.vertices),
                                  o3d.utility.Vector3iVector(ref.faces))))

    # Open reference surfaces (no thickness) get their sign from the nearest face's normal,
    # with normals turned to face away from the limb's axis, so + still means "outside the limb".
    face_n = ref.face_normals.copy()
    if not ref.is_watertight:
        cr0, ar0 = pca_frame(ref_pts)
        rel = ref.triangles_center - cr0
        radial = rel - np.outer(rel @ ar0[0], ar0[0])
        if np.mean(np.einsum("ij,ij->i", face_n, radial) < 0) > 0.5:
            face_n = -face_n

    def signed(pts):
        q = o3d.core.Tensor(pts.astype(np.float32))
        if ref.is_watertight:
            return scene.compute_signed_distance(q).numpy()
        cp = scene.compute_closest_points(q)
        diff = pts - cp["points"].numpy()
        sign = np.sign(np.einsum("ij,ij->i", diff, face_n[cp["primitive_ids"].numpy()]))
        return np.linalg.norm(diff, axis=1) * np.where(sign == 0, 1, sign)

    if not args.no_crop:
        # Only score the part of the test that lies along the reference's length (open
        # references have no caps, and the scan base can hold leftover table points).
        cr0, ar0 = pca_frame(ref_pts)
        tr = (ref_pts - cr0) @ ar0[0]
        tt = (test - cr0) @ ar0[0]
        test = test[(tt > tr.min() + args.crop_margin) & (tt < tr.max() - args.crop_margin)]

    d = signed(test)
    ad = np.abs(d)

    # Uniform scale error (e.g. print shrinkage or scanner scale bias). s > 1 means the scan is too big.
    c = test.mean(0)
    fit = minimize_scalar(lambda s: np.mean(np.abs(signed(c + (test - c) / s)) ** 2),
                          bounds=(0.8, 1.2), method="bounded")

    report = {
        "points": int(len(test)),
        "mean_abs_mm": float(ad.mean()), "median_abs_mm": float(np.median(ad)),
        "p95_abs_mm": float(np.percentile(ad, 95)), "max_abs_mm": float(ad.max()),
        "within_1mm_pct": float((ad <= 1).mean() * 100), "within_2mm_pct": float((ad <= 2).mean() * 100),
        "scale_factor": float(fit.x), "scale_error_pct": float((fit.x - 1) * 100),
        "mean_signed_mm": float(d.mean()),  # >0: test sits outside the true surface on average
        "median_signed_mm": float(np.median(d)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # colored points for viewing
    cmap = plt.get_cmap("turbo")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(test))
    pc.colors = o3d.utility.Vector3dVector(cmap(np.clip(ad / max(report["p95_abs_mm"], 1e-6), 0, 1))[:, :3])
    o3d.io.write_point_cloud(str(out / "error_colored.ply"), pc)

    # plots
    cr, ar = pca_frame(ref_pts)
    axis = ar[0]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].hist(d, bins=80, color="#4a7bd0")
    ax[0].set(title="Distance to true surface", xlabel="mm (+ = outside the true surface)", ylabel="points")
    t = (test - cr) @ axis
    ax[1].scatter(t, d, s=0.5, alpha=0.4)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(title="Error along the limb", xlabel="position along limb axis (mm)", ylabel="error (mm)")
    sec = ref.section(plane_origin=cr, plane_normal=axis)
    if sec is not None:
        for e in sec.discrete:
            ax[2].plot(*((e - cr) @ ar[1:].T).T, "k-", lw=1.2, label="true")
    band = np.abs(t) < 1.5
    s2 = (test[band] - cr) @ ar[1:].T
    ax[2].scatter(s2[:, 0], s2[:, 1], s=3, c="#d0504a", label="test")
    ax[2].set_aspect("equal")
    ax[2].set(title="Cross-section at mid-length", xlabel="mm", ylabel="mm")
    h, l = ax[2].get_legend_handles_labels()
    ax[2].legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
    fig.suptitle(f"mean {report['mean_abs_mm']:.2f} mm | p95 {report['p95_abs_mm']:.2f} mm | "
                 f"max {report['max_abs_mm']:.2f} mm | scale {report['scale_error_pct']:+.2f}%")
    fig.tight_layout()
    fig.savefig(out / "error_report.png", dpi=110)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
