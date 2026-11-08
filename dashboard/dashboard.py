"""
Web viewer for recorded trajectories (http://localhost:8000):

    python dashboard/dashboard.py [--traj-dir dashboard/trajectories] [--geometry-dir <bolt geometry>] [--port 8000]

Trajectories are read from --traj-dir, which training writes to (one subdirectory per experiment). Meshes are served
from bolt's geometry directory, or --geometry-dir when bolt is not installed. The page (js/viewer.js) fetches a
trajectory from /api/trajectory and plays it back in the browser.
"""
import argparse
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import Response
from nicegui import ui, app

DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASHBOARD_DIR.parent))  # for msk_envs, when it is not installed

from msk_envs.trajectory_io import dumps, is_current_format, load_animation  # noqa: E402


def default_geometry_dir() -> str:
    """
    bolt's geometry directory, from bolt/paths.py without importing bolt itself (which loads opensim and warp,
    and fails in nicegui's reload process, which starts without the parent's LD_LIBRARY_PATH)
    """
    spec = importlib.util.find_spec("bolt")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("Could not find bolt to locate its geometry directory; pass --geometry-dir")
    paths_file = Path(spec.submodule_search_locations[0]) / "paths.py"
    if paths_file.exists():
        paths_spec = importlib.util.spec_from_file_location("bolt_paths", paths_file)
        paths = importlib.util.module_from_spec(paths_spec)
        paths_spec.loader.exec_module(paths)
        return paths.get_geometry_dir()
    from bolt import get_geometry_dir  # older bolt versions
    return get_geometry_dir()


parser = argparse.ArgumentParser(description="MSK Dashboard")
parser.add_argument("--traj-dir", type=str, default=str(DASHBOARD_DIR / "trajectories"),
                    help="Trajectory directory (default: dashboard/trajectories)")
parser.add_argument("--geometry-dir", type=str, default=None,
                    help="Mesh directory (default: bolt's geometry directory)")
parser.add_argument("--port", type=int, default=8000)
args, _ = parser.parse_known_args()
TRAJ_DIR = Path(args.traj_dir).resolve()
GEOMETRY_DIR = args.geometry_dir or default_geometry_dir()


@app.get("/api/trajectory")
def get_trajectory(path: str):
    """ The trajectory at [path] (relative to the trajectory directory), as a gzipped version 2 animation """
    file = (TRAJ_DIR / path).resolve()
    if TRAJ_DIR not in file.parents or not file.is_file():
        raise HTTPException(status_code=404, detail=f"No trajectory {path}")
    if file.name.endswith(".gz") and is_current_format(str(file)):
        body = file.read_bytes()  # sent as is
    else:
        body = gzip.compress(dumps(load_animation(str(file))).encode(), compresslevel=3)  # older ones are converted
    return Response(content=body, media_type="application/json",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"})


def _matches_dir_filter(dir_name: str, filter_text: str) -> bool:
    if not filter_text:
        return True
    pattern = re.escape(filter_text).replace(r'\*', '.*')
    return re.search(pattern, dir_name, re.IGNORECASE) is not None


def _get_traj_files(base_path: Path) -> list[Path]:
    """ .json and .json.gz files under [base_path], by directory, then by step number in the file name """
    traj_files = list(base_path.rglob("*.json")) + list(base_path.rglob("*.json.gz"))

    def sort_key(path: Path):
        file_name = path.name.removesuffix(".gz").removesuffix(".json")
        numbers = re.findall(r'\d+', file_name)
        return str(path.parent), int(numbers[0]) if numbers else 0, file_name

    return sorted(traj_files, key=sort_key)


def _selected_base_path() -> Path:
    return TRAJ_DIR / selected_traj_dir if selected_traj_dir else TRAJ_DIR


def locate_traj_dirs():
    traj_dir_options.clear()
    dirs = sorted(d.name for d in TRAJ_DIR.iterdir() if d.is_dir()) if TRAJ_DIR.is_dir() else []
    filter_text = dir_filter_input.value.strip()

    with traj_dir_options:
        ui.item("All", on_click=lambda: select_traj_dir(""))
        for dir_name in dirs:
            if _matches_dir_filter(dir_name, filter_text):
                ui.item(dir_name, on_click=lambda d=dir_name: select_traj_dir(d))


def select_traj_dir(dir_name: str):
    """ Select a trajectory directory, and show its latest trajectory """
    global selected_traj_dir
    selected_traj_dir = dir_name
    traj_dir_label.set_text(f"Dir: {dir_name if dir_name else 'All'}")
    traj_files = _get_traj_files(_selected_base_path())
    if traj_files:
        send_viewer_trajectory(traj_files[-1])


def locate_trajectories():
    trajectory_options.clear()
    with trajectory_options:
        for path in _get_traj_files(_selected_base_path()):
            # With "All" selected, show the path relative to TRAJ_DIR
            name = str(path.relative_to(TRAJ_DIR)) if selected_traj_dir == "" else path.name
            ui.item(name, on_click=lambda p=path: send_viewer_trajectory(p))


def send_viewer_trajectory(path: Path):
    rel = str(path.resolve().relative_to(TRAJ_DIR))
    load_args = json.dumps([f"/api/trajectory?path={quote(rel)}", rel])
    ui.run_javascript(f"window.mskViewer ? window.mskViewer.load(...{load_args}) : (window.mskPendingLoad = {load_args})")


app.add_static_files('/css', DASHBOARD_DIR / 'css', max_cache_age=0)  # edits show up on reload
app.add_static_files('/js', DASHBOARD_DIR / 'js', max_cache_age=0)
app.add_static_files('/geometry', GEOMETRY_DIR)
app.add_static_files('/assets', DASHBOARD_DIR / 'assets')

VIEWER_HTML = """
<div id="viewerWrap">
  <div id="viewer"></div>
  <div id="loadingStatus" class="overlay-status">Pick a trajectory above</div>
  <div id="muscleTooltip"></div>

  <details id="sidePanel" class="panel" open><summary class="panel-title">Controls</summary>
    <details open><summary>Show</summary>
      <label><input type="checkbox" id="drawVisuals" checked> Bones</label>
      <label class="slider">opacity <input type="range" id="boneOpacity" min="0.1" max="1" step="0.05" value="1"></label>
      <label><input type="checkbox" id="drawMuscles" checked> Muscles</label>
      <label><input type="checkbox" id="drawWrap"> Wrap surfaces <span id="wrapCount"></span></label>
      <label><input type="checkbox" id="drawColliders"> Contact geometry</label>
      <label><input type="checkbox" id="drawBeams" checked> Beams</label>
      <label><input type="checkbox" id="drawGuides" checked> Command arrows</label>
      <div id="guideLegend"></div>
      <label><input type="checkbox" id="drawForces" checked> External forces</label>
      <label><input type="checkbox" id="drawShadows" checked> Shadows</label>
    </details>
    <details open><summary>Muscles</summary>
      <label class="slider">style
        <select id="muscleStyle">
          <option value="tubes">tubes</option>
          <option value="capsules" selected>capsules</option>
          <option value="lines">lines</option>
        </select></label>
      <label class="slider">color
        <select id="muscleColor">
          <option value="activation">activation</option>
          <option value="excitation">excitation</option>
          <option value="force">force / max</option>
        </select></label>
      <div class="colorbar-row"><span>0</span><div id="colorBar"></div><span>1</span></div>
      <label class="slider">thickness <input type="range" id="muscleScale" min="0.2" max="2.5" step="0.05" value="1"></label>
      <label><input type="checkbox" id="drawTendons" checked> Tendons</label>
      <input type="text" id="muscleFilter" placeholder="only: e.g. vas* glut*_r">
    </details>
    <details open><summary>Camera</summary>
      <label class="slider">view
        <select id="layout">
          <option value="single">single</option>
          <option value="panel">3 views</option>
          <option value="feet">with feet</option>
        </select></label>
      <label><input type="checkbox" id="follow" checked> Follow</label>
      <button id="resetCamera">Reset cameras</button>
    </details>
    <details open><summary>Scene</summary>
      <label class="slider">ground <select id="groundStyle"></select></label>
      <label class="slider">grid <select id="groundGrid"></select></label>
    </details>
  </details>

  <div id="playbar" class="panel">
    <button id="stepBack" title="Previous frame (left arrow)">◀︎◀︎</button>
    <button id="playButton" title="Play / pause (space)">▶</button>
    <button id="stepForward" title="Next frame (right arrow)">▶︎▶︎</button>
    <input type="range" id="timeline" min="0" max="0" value="0" step="1">
    <span id="timeValue">t = 0.000 s</span>
    <span id="frameValue"></span>
    <select id="speed" title="Playback speed">
      <option value="0.1">0.1x</option><option value="0.25">0.25x</option><option value="0.5">0.5x</option>
      <option value="1" selected>1x</option><option value="2">2x</option>
    </select>
    <label><input type="checkbox" id="loop" checked> loop</label>
  </div>
</div>
"""

ui.add_head_html("""
<script type="importmap">
    {
      "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.179.1/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/"
      }
    }
</script>
<script type="module" src="js/viewer.js"></script>
<link rel="stylesheet" href="css/main.css">
""")
ui.colors(primary='#555')

selected_traj_dir = ""

with ui.row().classes('w-full items-center gap-4 px-4 py-2 bg-white').style('border-bottom: 1px solid #ddd'):
    ui.label("MSK Dashboard").classes("text-xl font-bold")
    dir_filter_input = ui.input(label="Filter dir", placeholder="e.g. exp*", on_change=locate_traj_dirs) \
        .props("dense").classes("w-40")
    traj_dir_options = ui.dropdown_button("Traj Dir", auto_close=True, on_click=locate_traj_dirs).props("dense")
    traj_dir_label = ui.label("Dir: All")
    trajectory_options = ui.dropdown_button("Trajectories", auto_close=True, on_click=locate_trajectories) \
        .props("dense")
    with ui.column().classes("gap-0"):
        ui.html('<span id="currentTrajectory" class="font-mono text-sm"></span>')
        ui.html('<span id="trajectoryInfo" class="text-xs text-gray-500"></span>')

ui.html(VIEWER_HTML).classes('w-full')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(show=False, title="MSK Dashboard", favicon=DASHBOARD_DIR / "assets/textures/favicon.png", port=args.port)
