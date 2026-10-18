"""
Project settings (v2). Things a normal user should not have to type in the UI.
Override any of these with environment variables if needed.
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.environ.get("ROAD_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "best.pt"))
RUNS_DIR = os.environ.get("ROAD_RUNS_DIR", os.path.join(PROJECT_ROOT, "outputs", "runs"))

# Detection presets shown in the dashboard instead of a raw confidence slider
CONFIDENCE_PRESETS = {
    "High recall (0.25)": 0.25,
    "Balanced (0.40)": 0.40,
    "High precision (0.60)": 0.60,
}
DEFAULT_PRESET = "Balanced (0.40)"

DEFAULT_FRAME_INTERVAL = 5
SCENE_CUT_SENSITIVITY = 4.0   # higher = fewer cuts
VERIFY_MAX_MISSED_FRAMES = 1
VERIFY_MIN_STREAK = 2
