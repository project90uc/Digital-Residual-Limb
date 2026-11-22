"""Loading point clouds from .ply, .xyz, .txt and .csv files."""
from pathlib import Path

import numpy as np
import open3d as o3d

UNIT_TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}


def load_points(path, units="m", min_confidence=None):
    """Return an Open3D point cloud in millimetres.

    An optional 4th column is treated as per-point confidence (used by min_confidence).
    `units` is the unit of the file.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".ply":
        arr = _read_ply_ascii_or_binary(path)
    elif ext in (".xyz", ".txt", ".csv", ".pts"):
        arr = _read_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if arr.shape[1] < 3:
        raise ValueError(f"{path} has fewer than 3 numeric columns")
    if min_confidence is not None and arr.shape[1] >= 4:
        arr = arr[arr[:, 3] >= min_confidence]

    xyz = arr[:, :3] * UNIT_TO_MM[units]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return pc


def _read_text(path):
    delim = "," if path.suffix.lower() == ".csv" else None
    with open(path) as f:
        first = f.readline()
    skip = 0 if _is_numeric_row(first, delim) else 1  # skip header like "x,y,z,confidence"
    return np.atleast_2d(np.loadtxt(path, delimiter=delim, skiprows=skip))


def _is_numeric_row(line, delim):
    try:
        [float(v) for v in line.strip().split(delim) if v != ""]
        return True
    except ValueError:
        return False


def _read_ply_ascii_or_binary(path):
    """Read x, y, z (+ confidence if present). Falls back to Open3D for binary files."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line == "end_header" or not line:
                break
    is_ascii = any(h.startswith("format ascii") for h in header)
    props = [h.split()[-1] for h in header if h.startswith("property") and "list" not in h]
    n_vert = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
    if is_ascii:
        data = np.loadtxt(path, skiprows=len(header), max_rows=n_vert)
        data = np.atleast_2d(data)
        cols = [props.index(c) for c in ("x", "y", "z")]
        if "confidence" in props:
            cols.append(props.index("confidence"))
        return data[:, cols]
    pc = o3d.io.read_point_cloud(str(path))
    return np.asarray(pc.points)
