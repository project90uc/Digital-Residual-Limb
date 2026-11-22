"""Make a synthetic test case with a known answer.

Writes:
  reference_limb.stl   the "true" limb, in mm (what the cast was made from)
  fake_scan.ply        points on the INSIDE of the negative cast, as a scanner would
                       see them: metres, y up, sitting on a table, with noise,
                       outliers, no points across the open end.
"""
import argparse

import numpy as np
import trimesh


def make_limb(length=100.0, r_distal=28.0, r_proximal=40.0, bump=3.0):
    t = np.linspace(0, 1, 60)
    r = r_distal + (r_proximal - r_distal) * t ** 1.5
    z = t * length
    # rounded distal end
    a = np.linspace(-np.pi / 2, 0, 20, endpoint=False)
    cap_r = r_distal * np.cos(a)
    cap_z = r_distal * np.sin(a) * 0.8
    profile = np.column_stack([np.r_[0, cap_r[1:], r], np.r_[cap_z[0], cap_z[1:], z]])
    limb = trimesh.creation.revolve(profile, sections=128)
    limb = limb.subdivide().subdivide()
    # a bony prominence: exactly the kind of feature that matters for pressure
    v = limb.vertices
    center = np.array([r_distal + 5, 0, length * 0.35])
    d = np.linalg.norm(v - center, axis=1)
    radial = v[:, :2] / np.maximum(np.linalg.norm(v[:, :2], axis=1, keepdims=True), 1e-9)
    v[:, :2] += radial * (bump * np.exp(-(d / 10.0) ** 2))[:, None]
    limb.vertices = v
    # close the proximal end so the reference is a solid
    limb = trimesh.Trimesh(limb.vertices, limb.faces).slice_plane([0, 0, length - 0.01], [0, 0, -1], cap=True)
    return limb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, default=1.5, help="mm, std of scan noise")
    p.add_argument("--points", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    limb = make_limb()
    limb.export("reference_limb.stl")

    # Points on the limb surface = inner surface of the cast. Skip the open end (top cap).
    pts, fi = trimesh.sample.sample_surface(limb, args.points, seed=args.seed)
    keep = limb.face_normals[fi][:, 2] < 0.95
    pts = pts[keep]
    pts += rng.normal(0, args.noise, pts.shape)
    outliers = pts[rng.choice(len(pts), len(pts) // 100)] + rng.normal(0, 15, (len(pts) // 100, 3))

    # Cast sits on a table: distal end down, wall ~6 mm, so cast floor is above the table.
    floor_z = limb.bounds[0, 2] - 6.0
    ang = rng.uniform(0, 2 * np.pi, 15000)
    rad = rng.uniform(55, 160, 15000)
    table = np.column_stack([rad * np.cos(ang), rad * np.sin(ang), np.full(15000, floor_z)])
    table[:, 2] += rng.normal(0, args.noise, 15000)

    cloud = np.vstack([pts, outliers, table])
    # Into a typical scanner frame: metres, y up, arbitrary position and heading.
    zup_to_yup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    yaw = np.radians(37)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    cloud = (cloud @ zup_to_yup.T @ ry.T) / 1000.0 + np.array([0.12, -0.18, -0.30])
    conf = np.full((len(cloud), 1), 2.0)
    data = np.hstack([cloud, conf])

    header = ("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
              "property float z\nproperty float confidence\nend_header\n" % len(data))
    with open("fake_scan.ply", "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.6f")
    print(f"reference_limb.stl: {np.round(limb.extents, 1)} mm, volume {limb.volume / 1000:.1f} cm3")
    print(f"fake_scan.ply: {len(data)} points, noise {args.noise} mm")


if __name__ == "__main__":
    main()
