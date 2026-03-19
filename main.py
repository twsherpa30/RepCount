"""RepCount — Real-time exercise rep counter using MediaPipe Pose estimation.

Uses the MediaPipe Tasks API (PoseLandmarker) for pose detection.

Usage:
    python main.py                                          # default webcam + squat
    python main.py --camera 1                               # camera at index 1
    python main.py --camera-url http://192.168.1.5:4747/video  # IP camera (phone)
    python main.py --exercise bicep_curl                    # different exercise

Controls:
    Q — quit
    R — reset rep count
"""

import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

from utils import calculate_angle_3d
from rep_counter import RepCounter
from validator import check_visibility, is_side_on, is_body_in_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_EXERCISE = "squat"

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
        "down_threshold": 40,
        "up_threshold": 140,
        "joint_label": "Elbow",
    },
    "shoulder_press": {
        "landmarks": (13, 11, 23),  # elbow, shoulder, hip
        "down_threshold": 90,
        "up_threshold": 160,
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
        "down_threshold": 30,
        "up_threshold": 80,
        "joint_label": "Shoulder",
    },
}

# Path to the downloaded pose landmarker model
MODEL_PATH = "pose_landmarker.task"

# ---------------------------------------------------------------------------
# HUD drawing helpers
# ---------------------------------------------------------------------------
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_X = 15
HUD_BG_COLOR = (0, 0, 0)
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def _put(frame, text, y, color=WHITE, scale=0.6, thickness=1):
    """Shorthand for cv2.putText at a fixed x offset."""
    cv2.putText(frame, text, (HUD_X, y), HUD_FONT, scale, color, thickness,
                cv2.LINE_AA)


def draw_hud(frame, exercise, count, stage, warnings):
    """Draw the heads-up display overlay on the frame.

    Args:
        frame:    BGR image to draw on (mutated in-place).
        exercise: Current exercise name string.
        count:    Current rep count.
        stage:    Current stage string ("up", "down", or None).
        warnings: List of (message, color) tuples from validator checks.
    """
    # Background rectangle
    h = 90 + 28 * len(warnings)
    cv2.rectangle(frame, (0, 0), (400, h), HUD_BG_COLOR, -1)

    # Exercise / reps / stage
    display_name = exercise.replace("_", " ").upper()
    _put(frame, f"Exercise: {display_name}", 28, WHITE, 0.7, 2)
    _put(frame, f"Reps: {count}", 58, WHITE, 0.65, 2)
    cv2.putText(frame, f"Stage: {stage.upper() if stage else '--'}",
                (210, 58), HUD_FONT, 0.65, WHITE, 2, cv2.LINE_AA)

    # Separator line
    cv2.line(frame, (5, 70), (395, 70), (80, 80, 80), 1)

    # Warnings / status
    if not warnings:
        _put(frame, "Position: Good", 92, GREEN, 0.55, 1)
    else:
        y = 92
        for msg, color in warnings:
            _put(frame, msg, y, color, 0.55, 1)
            y += 28


def draw_angle_text(frame, angle, landmark, w, h):
    """Draw the current joint angle next to the landmark on the frame."""
    x_px = int(landmark.x * w)
    y_px = int(landmark.y * h)
    cv2.putText(frame, f"{int(angle)} deg", (x_px + 10, y_px),
                HUD_FONT, 0.5, (255, 255, 0), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Camera initialisation
# ---------------------------------------------------------------------------
def open_camera(camera_index=None, camera_url=None):
    """Open a video capture source.

    Priority: camera_url > camera_index > auto-detect (indices 0-2).
    """
    if camera_url:
        cap = cv2.VideoCapture(camera_url)
        if cap.isOpened():
            print(f"Camera opened on URL: {camera_url}")
            return cap
        raise RuntimeError(f"Cannot open camera at URL: {camera_url}")

    if camera_index is not None:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera opened on index {camera_index}")
            return cap
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened on index {idx}")
            return cap
    raise RuntimeError("No camera found on indices 0, 1, or 2.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RepCount — real-time rep counter")
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index to use (e.g. 0 for built-in webcam, 1 for iPhone). "
             "If omitted, indices 0-2 are tried automatically.",
    )
    parser.add_argument(
        "--camera-url", type=str, default=None,
        help="URL for an IP camera stream (e.g. http://192.168.1.5:4747/video). "
             "Takes priority over --camera.",
    )
    parser.add_argument(
        "--exercise", type=str, default=None,
        choices=list(EXERCISE_CONFIG.keys()),
        help="Exercise to track. If omitted, an interactive menu is shown.",
    )
    return parser.parse_args()


def select_exercise():
    """Display an interactive menu for exercise selection."""
    exercises = list(EXERCISE_CONFIG.keys())
    print("\n" + "=" * 40)
    print("  RepCount — Exercise Selection")
    print("=" * 40)
    for i, name in enumerate(exercises, 1):
        display = name.replace("_", " ").title()
        joint = EXERCISE_CONFIG[name]["joint_label"]
        print(f"  {i}. {display}  ({joint} angle)")
    print("=" * 40)

    while True:
        try:
            choice = input(f"\nSelect exercise [1-{len(exercises)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(exercises):
                selected = exercises[idx]
                print(f"\n→ Selected: {selected.replace('_', ' ').title()}\n")
                return selected
        except (ValueError, EOFError):
            pass
        print(f"  Invalid choice. Enter a number between 1 and {len(exercises)}.")


def main():
    args = parse_args()
    exercise = args.exercise if args.exercise else select_exercise()
    config = EXERCISE_CONFIG[exercise]
    lm_a, lm_b, lm_c = config["landmarks"]
    counter = RepCounter(config["down_threshold"], config["up_threshold"])

    cap = open_camera(args.camera, args.camera_url)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=os.path.join(os.path.dirname(__file__), MODEL_PATH)
    )
    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # Pose connections for skeleton drawing
    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_idx += 1

        # --- Pose detection via Tasks API ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        warnings = []
        can_count = True  # only hard-block on visibility

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]  # first detected pose

            # --- Validator checks ---
            low_vis = check_visibility(landmarks, exercise)
            side_on = is_side_on(landmarks)
            in_frame = is_body_in_frame(landmarks)

            if low_vis:
                warnings.append(("Counting paused -- body not visible", RED))
                can_count = False  # unreliable pose data
            if not side_on:
                warnings.append(("Tip: Turn sideways for best accuracy", ORANGE))
            if not in_frame:
                warnings.append(("Tip: Step back -- body near edge", ORANGE))

            # --- 3D Angle calculation ---
            a = [landmarks[lm_a].x, landmarks[lm_a].y, landmarks[lm_a].z]
            b = [landmarks[lm_b].x, landmarks[lm_b].y, landmarks[lm_b].z]
            c = [landmarks[lm_c].x, landmarks[lm_c].y, landmarks[lm_c].z]
            angle = calculate_angle_3d(a, b, c)

            # Count reps unless visibility is too low
            if can_count:
                counter.update(angle)

            # --- Draw skeleton ---
            drawing_utils.draw_landmarks(
                frame, landmarks, pose_connections
            )

            # --- Draw angle text ---
            draw_angle_text(frame, angle, landmarks[lm_b], w, h)
        else:
            warnings.append(("No pose detected", RED))

        # --- HUD ---
        draw_hud(frame, exercise, counter.count, counter.stage, warnings)

        cv2.imshow("RepCount", frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counter.reset()
            print("Rep counter reset.")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
