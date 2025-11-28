import os

# Absolute path to repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Static assets (checked into repo)
STATIC_ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
STATIC_MUSIC_DIR = os.path.join(STATIC_ASSETS_DIR, "music")

# Runtime storage defaults to /tmp on Spaces but can be overridden
RUNTIME_ROOT = os.environ.get("DIRECTORS_CUT_RUNTIME_DIR",
                              "/tmp/directors-cut")

INPUT_DIR = os.path.join(RUNTIME_ROOT, "input")
OUTPUT_DIR = os.path.join(RUNTIME_ROOT, "output")
ASSETS_TEMP_DIR = os.path.join(RUNTIME_ROOT, "assets-temp")
WORK_DIR = os.path.join(RUNTIME_ROOT, "work")
FRAMES_DIR = os.path.join(WORK_DIR, "frames")


def ensure_runtime_dirs() -> None:
    """Create all runtime directories if they do not exist."""
    for path in [INPUT_DIR, OUTPUT_DIR, ASSETS_TEMP_DIR, WORK_DIR, FRAMES_DIR]:
        os.makedirs(path, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "STATIC_ASSETS_DIR",
    "STATIC_MUSIC_DIR",
    "RUNTIME_ROOT",
    "INPUT_DIR",
    "OUTPUT_DIR",
    "ASSETS_TEMP_DIR",
    "WORK_DIR",
    "FRAMES_DIR",
    "ensure_runtime_dirs",
]

