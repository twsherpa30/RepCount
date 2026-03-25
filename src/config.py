"""Configuration constants for RepCount."""

import os

import cv2

# ---------------------------------------------------------------------------
# Exercise definitions
# ---------------------------------------------------------------------------
EXERCISE_CONFIG = {
    "squat": {
        "landmarks": (23, 25, 27),  # hip, knee, ankle
        "down_threshold": 90,
        "up_threshold": 160,
        "joint_label": "Knee",
    },
    "pushup": {
        "landmarks": (11, 13, 15),  # shoulder, elbow, wrist
        "down_threshold": 90,
        "up_threshold": 160,
        "joint_label": "Elbow",
    },
    "bicep_curl": {
        "landmarks": (11, 13, 15),  # shoulder, elbow, wrist
        "down_threshold": 60,
        "up_threshold": 140,
        "joint_label": "Elbow",
    },
    "shoulder_press": {
        "landmarks": (13, 11, 23),  # elbow, shoulder, hip
        "down_threshold": 90,
        "up_threshold": 150,
        "joint_label": "Shoulder",
    },
    "deadlift": {
        "landmarks": (11, 23, 25),  # shoulder, hip, knee
        "down_threshold": 90,
        "up_threshold": 160,
        "joint_label": "Hip",
    },
    "lunge": {
        "landmarks": (23, 25, 27),  # hip, knee, ankle
        "down_threshold": 90,
        "up_threshold": 160,
        "joint_label": "Knee",
    },
    "lateral_raise": {
        "landmarks": (23, 11, 13),  # hip, shoulder, elbow
        "down_threshold": 25,
        "up_threshold": 60,
        "joint_label": "Shoulder",
    },
    "bench_press": {
        "landmarks": (11, 13, 15),  # shoulder, elbow, wrist
        "down_threshold": 90,
        "up_threshold": 160,
        "joint_label": "Elbow",
    },
    "leg_press": {
        "landmarks": (23, 25, 27),  # hip, knee, ankle
        "down_threshold": 90,
        "up_threshold": 160,
        "joint_label": "Knee",
    },
    "pullup": {
        "landmarks": (11, 13, 15),  # shoulder, elbow, wrist
        "down_threshold": 80,
        "up_threshold": 150,
        "joint_label": "Elbow",
    },
}

EXERCISE_LIST = list(EXERCISE_CONFIG.keys())

# Path to the downloaded pose landmarker model (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker.task")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_ALPHA = 0.70          # overlay transparency
ACCENT = (0, 220, 120)   # green accent
ACCENT_DIM = (0, 140, 80)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
DARK = (30, 30, 30)
STAGE_UP_COLOR = (0, 220, 120)
STAGE_DOWN_COLOR = (0, 165, 255)

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_ITALIC = cv2.FONT_HERSHEY_SIMPLEX
