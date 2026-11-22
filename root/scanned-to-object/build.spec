# PyInstaller build settings for the desktop app.
#   pip install -r requirements.txt pyinstaller
#   pyinstaller build.spec
# Output: dist/ScanToLimbModel/ (Windows/Linux folder) or dist/ScanToLimbModel.app (macOS)
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

APP = "ScanToLimbModel"

# Open3D has no built-in PyInstaller hook. Collect it by hand, leaving out its
# machine-learning extras (they need PyTorch/TensorFlow, which this app doesn't use).
def _not_ml(name):
    return not any(part in name for part in (".ml", "_ml3d", "web_visualizer", "agent_skills"))

o3d_hidden = [m for m in collect_submodules("open3d") if _not_ml(m)]
o3d_bins = [b for b in collect_dynamic_libs("open3d") if "_ops." not in b[0]]
o3d_data = collect_data_files("open3d", excludes=["**/examples/**", "**/_ml3d/**", "**/ml/**",
                                                  "**/*_ops.*", "**/agent_skills/**"])

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=o3d_bins,
    datas=o3d_data,
    hiddenimports=o3d_hidden + ["pymeshfix", "mapbox_earcut", "shapely", "scipy.optimize"],
    excludes=["torch", "tensorflow", "IPython", "notebook", "jupyter", "ipywidgets",
              "sklearn", "PyQt5", "PySide6"],
    noarchive=False,
)
# Drop Open3D's TensorFlow/PyTorch operator libraries (large, and unused here).
a.binaries = [b for b in a.binaries if "_ops." not in b[0]]
a.datas = [d for d in a.datas if "_ops." not in d[0]]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP, console=False,
          upx=False, target_arch=None)
coll = COLLECT(exe, a.binaries, a.datas, name=APP, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(coll, name=f"{APP}.app", bundle_identifier="ca.project90.scantolimbmodel",
                 info_plist={"NSHighResolutionCapable": True})
