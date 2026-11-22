"""Desktop app: pick a scan, convert it to a limb model, preview it, save the STL.

Run from source:  python app.py
Packaged app:     see build.spec
"""
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline

APP_NAME = "Scan to Limb Model"
SCAN_TYPES = [("Point clouds", "*.ply *.csv *.txt *.xyz"), ("All files", "*.*")]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x680")
        self.minsize(900, 560)
        self.messages = queue.Queue()
        self.work_dir = None
        self.running = False
        self._build_ui()
        self.after(100, self._poll)

    # ------------------------------------------------------------ layout
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="1. Scan file", font=("", 11, "bold")).grid(row=0, column=0, sticky="w", **pad)
        self.scan_var = tk.StringVar()
        row = ttk.Frame(left)
        row.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Entry(row, textvariable=self.scan_var, width=34).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self._browse).pack(side="left", padx=(6, 0))

        ttk.Label(left, text="2. Settings", font=("", 11, "bold")).grid(row=2, column=0, sticky="w", **pad)
        settings = ttk.Frame(left)
        settings.grid(row=3, column=0, sticky="ew", **pad)

        ttk.Label(settings, text="File units").grid(row=0, column=0, sticky="w")
        self.units_var = tk.StringVar(value="m")
        ttk.Combobox(settings, textvariable=self.units_var, values=["m", "cm", "mm"], width=6,
                     state="readonly").grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(settings, text="Detail (mm)").grid(row=1, column=0, sticky="w")
        self.res_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(settings, from_=0.5, to=10, increment=0.5, textvariable=self.res_var,
                    width=6).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(settings, text="larger = smoother, for noisy scans",
                  foreground="#666").grid(row=1, column=2, sticky="w", padx=6)

        self.table_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="Scan includes a table or floor (remove it)",
                        variable=self.table_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        self.adv_shown = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Show advanced settings", variable=self.adv_shown,
                        command=self._toggle_advanced).grid(row=4, column=0, sticky="w", **pad)
        self.adv = ttk.Frame(left)
        self.conf_var = tk.StringVar(value="")
        self.eps_var = tk.DoubleVar(value=12.0)
        self.cap_var = tk.StringVar(value="auto")
        ttk.Label(self.adv, text="Min confidence").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.adv, textvariable=self.conf_var, width=6).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(self.adv, text="blank = keep all", foreground="#666").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Label(self.adv, text="Cluster gap (mm)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(self.adv, from_=2, to=50, increment=1, textvariable=self.eps_var,
                    width=6).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(self.adv, text="lower if other objects get included",
                  foreground="#666").grid(row=1, column=2, sticky="w", padx=6)
        ttk.Label(self.adv, text="Flat cap").grid(row=2, column=0, sticky="w")
        ttk.Combobox(self.adv, textvariable=self.cap_var, values=["auto", "top", "bottom", "both"],
                     width=6, state="readonly").grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(left, text="3. Convert", font=("", 11, "bold")).grid(row=6, column=0, sticky="w", **pad)
        self.convert_btn = ttk.Button(left, text="Convert to model", command=self._convert)
        self.convert_btn.grid(row=7, column=0, sticky="ew", **pad)
        self.progress = ttk.Progressbar(left, mode="determinate", maximum=6)
        self.progress.grid(row=8, column=0, sticky="ew", **pad)

        # Colors are set explicitly so the text stays readable in macOS/Windows dark mode.
        self.log = tk.Text(left, height=12, width=52, state="disabled", wrap="word",
                           font=("Courier", 11), background="#f6f6f6", foreground="#222222",
                           relief="flat")
        self.log.grid(row=9, column=0, sticky="nsew", **pad)
        left.rowconfigure(9, weight=1)

        btns = ttk.Frame(left)
        btns.grid(row=10, column=0, sticky="ew", **pad)
        self.save_btn = ttk.Button(btns, text="Save STL...", command=self._save, state="disabled")
        self.save_btn.pack(side="left", fill="x", expand=True)
        self.folder_btn = ttk.Button(btns, text="Show working files", command=self._open_folder,
                                     state="disabled")
        self.folder_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # preview
        self.summary = tk.Label(right, text="Load a scan and press Convert.", font=("", 12),
                                background="white", foreground="#222222", anchor="w", padx=8, pady=6)
        self.summary.pack(fill="x")
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._blank_preview()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(right, text="Drag to rotate the preview.", foreground="#666").pack(anchor="w")

    def _toggle_advanced(self):
        if self.adv_shown.get():
            self.adv.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        else:
            self.adv.grid_remove()

    def _blank_preview(self):
        self.ax.clear()
        self.ax.set_axis_off()

    # ------------------------------------------------------------ actions
    def _browse(self):
        path = filedialog.askopenfilename(title="Choose a scan", filetypes=SCAN_TYPES)
        if path:
            self.scan_var.set(path)

    def _convert(self):
        scan = self.scan_var.get().strip()
        if not scan or not Path(scan).is_file():
            messagebox.showerror(APP_NAME, "Choose a scan file first.")
            return
        argv = [scan, "-o", "", "--units", self.units_var.get(),
                "--resolution", str(self.res_var.get()), "--cluster-eps", str(self.eps_var.get()),
                "--cap", self.cap_var.get()]
        if not self.table_var.get():
            argv.append("--no-table")
        if self.conf_var.get().strip():
            argv += ["--min-confidence", self.conf_var.get().strip()]

        self.work_dir = Path(tempfile.mkdtemp(prefix="limb_model_"))
        argv[2] = str(self.work_dir)
        try:
            args = pipeline.build_parser().parse_args(argv)
        except SystemExit:
            messagebox.showerror(APP_NAME, "One of the settings is not valid.")
            return

        self.running = True
        self.convert_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.folder_btn.config(state="disabled")
        self.progress["value"] = 0
        self._log_clear()
        self.summary.config(text="Working...")
        self._blank_preview()
        self.canvas.draw_idle()
        threading.Thread(target=self._worker, args=(args,), daemon=True).start()

    def _worker(self, args):
        try:
            meta = pipeline.run(args, log=lambda line: self.messages.put(("log", line)))
            mesh = o3d.io.read_triangle_mesh(str(Path(args.out) / "06_solid.ply"))
            if len(mesh.triangles) > 8000:
                mesh = mesh.simplify_quadric_decimation(8000)
            self.messages.put(("done", (meta, np.asarray(mesh.vertices), np.asarray(mesh.triangles))))
        except Exception as e:  # show any failure to the user instead of a silent crash
            self.messages.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    if "/6]" in payload:  # "[2/6] isolate..." step start, shown to the user
                        self._log(payload)
                    elif payload.startswith("["):  # "[2] isolate {...}" step done
                        self.progress["value"] = int(payload[1])
                elif kind == "done":
                    self._finish(*payload)
                elif kind == "error":
                    self._fail(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, meta, verts, faces):
        self.running = False
        self.convert_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.folder_btn.config(state="normal")
        size = " x ".join(f"{v:.0f}" for v in meta.get("size_mm", []))
        vol = meta.get("volume_cm3")
        text = f"Size {size} mm"
        if vol:
            text += f"   |   Volume {vol:.0f} cm³"
        text += "   |   " + ("Watertight" if meta.get("watertight") else "NOT watertight")
        self.summary.config(text=text)
        if meta.get("warning"):
            self._log("Note: " + meta["warning"])
        self._draw(verts, faces)

    def _fail(self, msg):
        self.running = False
        self.convert_btn.config(state="normal")
        self.summary.config(text="Conversion failed.")
        self._log("ERROR: " + msg.splitlines()[0])
        messagebox.showerror(APP_NAME, "Conversion failed:\n\n" + msg)

    def _draw(self, v, f):
        self.ax.clear()
        c = v.mean(0)
        v = v - c
        self.ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#c9d4e3",
                             edgecolor="none", linewidth=0, shade=True)
        r = np.abs(v).max()
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_zlim(-r, r)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=15, azim=-60, vertical_axis="y")
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def _save(self):
        src = self.work_dir / "06_solid.stl"
        default = Path(self.scan_var.get()).stem + "_model.stl"
        dest = filedialog.asksaveasfilename(title="Save model", defaultextension=".stl",
                                            initialfile=default, filetypes=[("STL", "*.stl")])
        if dest:
            shutil.copyfile(src, dest)
            self._log(f"Saved {dest}")

    def _open_folder(self):
        path = str(self.work_dir)
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    # ------------------------------------------------------------ log
    def _log(self, line):
        self.log.config(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    app = App()
    # A scan passed on launch (e.g. a file dropped onto the app icon) is pre-filled.
    files = [p for p in sys.argv[1:] if Path(p).is_file()]
    if files:
        app.scan_var.set(files[0])
    app.mainloop()
