# scanned-to-object

Converts a point cloud of a negative cast into a watertight positive limb model (STL, millimetres), and measures its accuracy against a reference model.

Works with any scanner that outputs x, y, z points.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Desktop app
`app.py` is a simple window for non-technical users: choose a scan, press "Convert to model", check the 3D preview, then "Save STL". A few extra options (minimum confidence, cluster gap, which end to cap) are under "Show advanced settings".

Run from source:
```bash
python app.py
```
The app uses Tkinter, which comes with Python from python.org. On Linux, install it separately (`sudo apt install python3-tk`).

### Build the app
The app is packaged with PyInstaller. A build only runs on the operating system it was made on.
```bash
pip install pyinstaller
pyinstaller build.spec
```
Output: `dist/ScanToLimbModel/` on Windows (run `ScanToLimbModel.exe` inside), `dist/ScanToLimbModel.app` on macOS. Don't commit `build/` or `dist/`.

### Automatic builds (GitHub Actions)
`.github/workflows/build-app.yml` (at the repository root) builds the app on GitHub's Windows and macOS machines:
- Any time: Actions tab → *Build desktop app* → *Run workflow*. Download the zips from the run's Artifacts.
- For a release: push a version tag. The zips are attached to a GitHub Release.
  ```bash
  git tag v0.1.0
  git push origin v0.1.0
  ```

The builds aren't code-signed, so the first launch shows a security warning:
- Windows: "Windows protected your PC" → *More info* → *Run anyway*
- macOS: try to open the app once, then System Settings → Privacy & Security → *Open Anyway*

Signing removes these warnings, but it needs a paid certificate (Windows) or Apple Developer account (macOS).

## Input
- Formats: `.ply`, `.csv`, `.txt`, `.xyz`
- Columns: `x y z`, with an optional 4th `confidence` column. A header row is fine.
- Units: metres by default. Set `--units mm` or `--units cm` otherwise.
- The points should be the inside surface of the cast. That surface is the limb's shape.

## Usage
```bash
python pipeline.py <scan file> -o out/
```
The final model is `out/06_solid.stl`.

### Steps
Each step saves its result, so you can inspect or hand-edit any stage.

| # | Step | Output | What it does |
|---|---|---|---|
| 1 | load | `01_load.ply` | Reads the file and converts it to mm. Optionally filters by confidence. |
| 2 | isolate | `02_isolate.ply` | Removes the largest flat plane (e.g. a table) and everything below it, then keeps the largest cluster of points. |
| 3 | clean | `03_clean.ply` | Removes outlier points. |
| 4 | normals | `04_normals.ply` | Estimates surface normals and orients them outward from the limb's axis. |
| 5 | surface | `05_surface.ply` | Screened Poisson surface reconstruction. |
| 6 | solid | `06_solid.stl` | Keeps the largest piece, cuts away surface past the cast's opening, caps it flat, repairs holes if needed, and exports. |

`out/meta.json` records each step's results (point counts, axis, volume, size, whether it's watertight).

### Resume after a manual edit
Fix a file in any CAD or 3D software, then continue from the next step:
```bash
python pipeline.py out/03_clean.ply -o out/ --from 4
```

### Options
| Option | Default | Use |
|---|---|---|
| `--units m\|cm\|mm` | `m` | Units of the input file |
| `--resolution` | `2.0` | Surface detail in mm. Raise it for noisy scans (the pipeline warns when noise creates tunnels in the surface). |
| `--min-confidence` | off | Drop points below this confidence value |
| `--no-table` | off | Skip plane removal when the scan has no table or floor. The limb axis is then estimated from the point shape. |
| `--cluster-eps` | `12` | mm. Lower it if nearby objects get included with the cast. |
| `--outlier-std` | `2.0` | Lower = stricter outlier removal |
| `--voxel` | `0` | mm. Downsample dense scans to speed things up. |
| `--cap auto\|top\|bottom\|both` | `auto` | Which open end to cut flat. Auto picks the end with no data near the axis. |
| `--no-cap` | off | Keep Poisson's closed surface as-is |
| `--poissonrecon <path>` | off | Use the [mkazhdan/PoissonRecon](https://github.com/mkazhdan/PoissonRecon) binary instead of Open3D (`--point-weight` applies) |

Run `python pipeline.py --help` for every option.

## Accuracy check
```bash
python compare.py <test file> <reference file> -o compare_out/
```
- Test: a point cloud or mesh, for example `out/03_clean.ply` (raw scan) or `out/06_solid.stl` (final model).
- Reference: the true shape as a mesh. It can be closed or an open surface with no thickness.
- Both files are read as mm. Use `--test-units` / `--ref-units` for files in m or cm (e.g. `--test-units m` for a raw scan file).
- The two are aligned automatically (PCA, then ICP), so they don't need to share a coordinate frame. Use `--no-align` if they already do.
- Only the part of the test along the reference's length is scored. Use `--no-crop` to score everything.

Outputs:
- `report.json`: mean, median, p95 and max error (mm), % within 1 and 2 mm, fitted scale error. Positive signed error means the test lies outside the reference surface.
- `error_report.png`: error histogram, error along the limb, mid-length cross-section overlay
- `error_colored.ply`: test points colored by error

## Test without a scanner
```bash
python make_test_cast.py                     # writes fake_scan.ply + reference_limb.stl
python pipeline.py fake_scan.ply -o out/
python compare.py out/06_solid.stl reference_limb.stl
```
The synthetic case is a limb with a bony prominence, sampled from inside a cast, with 1.5 mm noise, 1% outliers and a table. The expected result is a watertight solid with a mean error of about 0.2 to 0.25 mm (it varies slightly between machines).

## AI assistance
This folder was developed with the help of Claude, an AI assistant made by Anthropic.

## Limitations
- All isolated points are treated as limb surface. A scan that also captures the cast's outer wall will blend both walls together.
- With table removal on, the cast is assumed to stand upright on the table. Use `--no-table` for other setups.
- Accuracy is bounded by scan noise. Poisson averages random noise, but it can't recover detail smaller than `--resolution`.
- `compare.py`'s automatic alignment finds the best fit, which slightly understates the true error.
