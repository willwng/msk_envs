"""
Web viewer for recorded trajectories (http://localhost:8000):

    python dashboard/dashboard.py [--traj-dir dashboard/trajectories] [--geometry-dir <bolt geometry>] [--port 8000]

Trajectories are read from --traj-dir, which training writes to (one subdirectory per experiment). Meshes are served
from bolt's geometry directory, or --geometry-dir when bolt is not installed.
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

from nicegui import ui, app, events

DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASHBOARD_DIR))  # for timeline_controller when imported from elsewhere
sys.path.insert(1, str(DASHBOARD_DIR.parent))  # for msk_envs, when it is not installed

from msk_envs.trajectory_io import load_trajectory, trajectory_fps  # noqa: E402
from timeline_controller import TimelineController  # noqa: E402


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
TRAJ_DIR = Path(args.traj_dir)
GEOMETRY_DIR = args.geometry_dir or default_geometry_dir()


def dispatch_custom_event(event_name: str, data: str):
    """ Helper for custom events to viewer or any listeners """
    ui.run_javascript(f'''
        window.dispatchEvent(new CustomEvent('{event_name}', {{
            detail: {data}
        }}));
    ''')


def format_timeline_label():
    slider_value_label.set_text(f"Frame {timeline_slider.value} of {timeline_slider.props['max']}")


def reset_dashboard():
    dispatch_custom_event('resetViewer', '{}')
    timeline_slider.value = 0
    format_timeline_label()


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
    reset_dashboard()
    traj = load_trajectory(str(path))
    if not traj:
        ui.notify(f"{path} has no frames")
        return

    # update slider
    current_traj_label.set_text(str(path))
    timeline_controller.max_ticks = len(traj) - 1
    timeline_slider.props['max'] = len(traj) - 1
    timeline_slider.update()
    format_timeline_label()

    # update playback rate to match the recording rate
    fps = trajectory_fps(traj)
    timeline_controller.set_tick_rate(fps)
    tick_rate_input.value = int(fps)
    tick_rate_input.update()

    dispatch_custom_event('loadTrajectory', traj)


def slider_change(e: events.ValueChangeEventArguments):
    timeline_controller.slider_change(e.value)
    format_timeline_label()
    dispatch_custom_event('sliderChanged', e.value)


app.add_static_files('/css', DASHBOARD_DIR / 'css')
app.add_static_files('/js', DASHBOARD_DIR / 'js')
app.add_static_files('/geometry', GEOMETRY_DIR)
app.add_static_files('/assets', DASHBOARD_DIR / 'assets')

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

with ui.row().classes('w-full z-50 bg-white').style(
        'border: 3px solid #ccc; border-radius: 8px; box-sizing: border-box;'):
    # Title and trajectory selection
    with ui.row().classes("w-full justify-center mt-4"):
        ui.label("MSK Dashboard").classes("text-2xl font-bold")
        with ui.row().classes("items-center justify-center gap-4"):
            dir_filter_input = ui.input(
                label="Filter dir",
                placeholder="e.g. exp*",
                on_change=locate_traj_dirs,
            ).classes("w-48")
            traj_dir_options = ui.dropdown_button("Traj Dir", auto_close=True, on_click=locate_traj_dirs)
            traj_dir_label = ui.label("Dir: All")
            trajectory_options = ui.dropdown_button("Trajectories", auto_close=True, on_click=locate_trajectories)

    # Renderer/viewer (see js/viewer.js)
    ui.html("""
        <div style="width: 100%">
            <div id="viewer"></div>
            <div class="controls" style="padding: 1em;">
                <div class="control-row">
                    <label for="drawVisuals">Draw visuals</label>
                    <input type="checkbox" id="drawVisuals" checked>

                    <label for="drawBeams">Draw beams</label>
                    <input type="checkbox" id="drawBeams" checked>

                    <label for="drawCapsuleColliders">Draw capsule colliders</label>
                    <input type="checkbox" id="drawCapsuleColliders">

                    <label for="drawSphereColliders">Draw sphere colliders</label>
                    <input type="checkbox" id="drawSphereColliders">

                    <label for="drawMuscles">Draw muscles</label>
                    <input type="checkbox" id="drawMuscles" checked>
                </div>
                <button id="fullScreenButton">Panel View</button>
                <div class="control-row">
                    <button id="resetButton1">Camera 1 follow</button>
                    <button id="resetButton2">Camera 2 follow</button>
                    <button id="resetButton3">Camera 3 follow</button>
                </div>
            </div>
        </div>
    """).classes('w-full')

    # Playback controls
    with ui.column().classes("items-center w-full"):
        timeline_slider = ui.slider(min=0, max=0, value=0, on_change=slider_change)
        timeline_slider.props('id="timeline"')
        with ui.row().classes("items-center"):
            current_traj_label = ui.label("")
            slider_value_label = ui.label("")
            time_value_label = ui.label("").props('id="timeValue"')
            play_button = ui.button("Play", on_click=lambda: timeline_controller.toggle_playback())
            ui.label("Tick Rate:")
            tick_rate_input = ui.number(
                value=15,
                min=0.0,
                step=1.0,
                on_change=lambda e: timeline_controller.set_tick_rate(e.value)
            ).classes("w-20")

timeline_controller = TimelineController(tick_rate=tick_rate_input.value,
                                         max_ticks=timeline_slider.props["max"],
                                         play_button=play_button,
                                         timeline_slider=timeline_slider)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(show=False, title="MSK Dashboard", favicon=DASHBOARD_DIR / "assets/textures/favicon.png", port=args.port)
