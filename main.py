"""RepCount — Real-time exercise rep counter using MediaPipe Pose estimation.

Uses the MediaPipe Tasks API (PoseLandmarker) for pose detection.

Usage:
    python main.py                                          # default webcam + menu
    python main.py --camera 1                               # camera at index 1
    python main.py --camera-url http://192.168.1.5:4747/video  # IP camera (phone)
    python main.py --exercise bicep_curl                    # skip menu, start directly

Controls:
    1-9, 0 — switch exercise on the fly
    R   — reset rep count
    Q   — quit
"""

import argparse
import math
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
        "down_threshold": 60,
        "up_threshold": 150,
        "joint_label": "Elbow",
    },
}

EXERCISE_LIST = list(EXERCISE_CONFIG.keys())

# Path to the downloaded pose landmarker model
MODEL_PATH = "pose_landmarker.task"

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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _overlay_rect(frame, x1, y1, x2, y2, color, alpha):
    """Draw a semi-transparent filled rectangle."""
    sub = frame[y1:y2, x1:x2]
    rect = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


def draw_exercise_bar(frame, active_exercise):
    """Draw the exercise selector bar at the bottom of the frame.

    Shows numbered exercises with the active one highlighted.
    """
    h, w = frame.shape[:2]
    bar_height = 40
    bar_y = h - bar_height

    # Semi-transparent background
    _overlay_rect(frame, 0, bar_y, w, h, DARK, 0.80)

    # Separator line
    cv2.line(frame, (0, bar_y), (w, bar_y), ACCENT_DIM, 1)

    # Calculate even spacing
    n = len(EXERCISE_LIST)
    col_w = w // n

    for i, name in enumerate(EXERCISE_LIST):
        x = i * col_w
        display = name.replace("_", " ").title()
        
        # Abbreviate to fit on screen if too many exercises
        if n > 7:
            display = display.replace("Shoulder", "Shldr").replace("Lateral", "Lat")
            if len(display) > 8:
                display = display[:6] + "."
                
        key_label = str((i + 1) % 10) if i < 10 else chr(ord('a') + i - 10)
        label = f"{key_label}: {display}"
        is_active = (name == active_exercise)

        # Highlight active exercise
        if is_active:
            _overlay_rect(frame, x, bar_y + 1, x + col_w, h, ACCENT, 0.25)
            color = ACCENT
            thickness = 2
        else:
            color = GRAY
            thickness = 1

        # Fit text within column
        font_scale = 0.40
        text_size = cv2.getTextSize(label, HUD_FONT, font_scale, thickness)[0]
        text_x = x + (col_w - text_size[0]) // 2
        text_y = bar_y + (bar_height + text_size[1]) // 2

        cv2.putText(frame, label, (text_x, text_y), HUD_FONT,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, exercise, count, stage, angle, config, warnings):
    """Draw the heads-up display overlay panel at top-left.

    Features a semi-transparent panel with rep count, exercise name,
    stage indicator, current angle, and warning messages.
    """
    # --- Panel dimensions ---
    panel_w = 340
    base_h = 150
    warn_h = 26 * len(warnings)
    panel_h = base_h + warn_h

    _overlay_rect(frame, 0, 0, panel_w, panel_h, DARK, BG_ALPHA)

    # Accent bar on the left edge
    cv2.rectangle(frame, (0, 0), (4, panel_h), ACCENT, -1)

    # --- Exercise name ---
    display_name = exercise.replace("_", " ").upper()
    cv2.putText(frame, display_name, (15, 30), HUD_FONT,
                0.70, WHITE, 2, cv2.LINE_AA)

    # --- Rep count (large) ---
    count_str = str(count)
    cv2.putText(frame, count_str, (15, 85), HUD_FONT,
                1.8, ACCENT, 3, cv2.LINE_AA)

    # "REPS" label next to count
    count_text_w = cv2.getTextSize(count_str, HUD_FONT, 1.8, 3)[0][0]
    cv2.putText(frame, "REPS", (20 + count_text_w, 85), HUD_FONT,
                0.50, GRAY, 1, cv2.LINE_AA)

    # --- Stage indicator ---
    stage_text = stage.upper() if stage else "--"
    if stage == "up":
        stage_color = STAGE_UP_COLOR
    elif stage == "down":
        stage_color = STAGE_DOWN_COLOR
    else:
        stage_color = GRAY

    # Stage pill background
    pill_x = 200
    pill_y = 58
    pill_w = 90
    pill_h = 32
    _overlay_rect(frame, pill_x, pill_y, pill_x + pill_w, pill_y + pill_h,
                  stage_color, 0.25)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                  stage_color, 1)
    text_size = cv2.getTextSize(stage_text, HUD_FONT, 0.55, 2)[0]
    text_x = pill_x + (pill_w - text_size[0]) // 2
    text_y = pill_y + (pill_h + text_size[1]) // 2
    cv2.putText(frame, stage_text, (text_x, text_y), HUD_FONT,
                0.55, stage_color, 2, cv2.LINE_AA)

    # --- Angle readout ---
    if angle is not None:
        angle_str = f"{int(angle)} deg"
        cv2.putText(frame, angle_str, (200, 30), HUD_FONT,
                    0.50, GRAY, 1, cv2.LINE_AA)

    # --- Joint label ---
    joint = config["joint_label"]
    cv2.putText(frame, f"Joint: {joint}", (15, 115), HUD_FONT,
                0.45, GRAY, 1, cv2.LINE_AA)

    # --- Separator ---
    cv2.line(frame, (10, 125), (panel_w - 10, 125), (80, 80, 80), 1)

    # --- Warnings / status ---
    if not warnings:
        cv2.putText(frame, "Position: Good", (15, 145), HUD_FONT,
                    0.50, ACCENT, 1, cv2.LINE_AA)
    else:
        y = 145
        for msg, color in warnings:
            cv2.putText(frame, msg, (15, y), HUD_FONT,
                        0.45, color, 1, cv2.LINE_AA)
            y += 26


def draw_angle_arc(frame, angle, config, landmark, w, h):
    """Draw a progress arc near the tracked joint.

    The arc fills based on where the current angle sits between
    the down and up thresholds.
    """
    down_t = config["down_threshold"]
    up_t = config["up_threshold"]

    # Normalise angle into 0–1 progress
    range_deg = up_t - down_t
    if range_deg == 0:
        progress = 0.0
    else:
        progress = max(0.0, min(1.0, (angle - down_t) / range_deg))

    # Position at the joint landmark
    cx = int(landmark.x * w) + 30
    cy = int(landmark.y * h)
    radius = 28

    # Background arc (full circle, dim)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 360,
                (60, 60, 60), 2, cv2.LINE_AA)

    # Progress arc
    end_angle = int(progress * 360)
    if progress < 0.3:
        arc_color = ORANGE
    elif progress > 0.85:
        arc_color = ACCENT
    else:
        arc_color = WHITE

    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle,
                arc_color, 3, cv2.LINE_AA)

    # Angle text in centre
    cv2.putText(frame, f"{int(angle)}", (cx - 14, cy + 5), HUD_FONT,
                0.40, arc_color, 1, cv2.LINE_AA)


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
        choices=EXERCISE_LIST,
        help="Exercise to track. If omitted, an interactive menu is shown.",
    )
    return parser.parse_args()


def select_exercise():
    """Display an interactive menu for exercise selection."""
    print("\n" + "=" * 40)
    print("  RepCount — Exercise Selection")
    print("=" * 40)
    for i, name in enumerate(EXERCISE_LIST, 1):
        display = name.replace("_", " ").title()
        joint = EXERCISE_CONFIG[name]["joint_label"]
        key_label = str(i % 10) if i <= 10 else chr(ord('a') + i - 11)
        print(f"  {key_label}. {display}  ({joint} angle)")
    print("=" * 40)

    while True:
        try:
            choice = input(f"\nSelect exercise: ").strip()
            if choice == "0":
                idx = 9
            elif choice.isdigit() and 1 <= int(choice) <= 9:
                idx = int(choice) - 1
            elif choice.isalpha():
                idx = 10 + ord(choice.lower()) - ord('a')
            else:
                idx = -1
                
            if 0 <= idx < len(EXERCISE_LIST):
                selected = EXERCISE_LIST[idx]
                print(f"\n→ Selected: {selected.replace('_', ' ').title()}\n")
                return selected
        except (ValueError, EOFError):
            pass
        print(f"  Invalid choice. Enter a valid key for the exercise.")


def switch_exercise(name):
    """Build fresh config, landmark indices, and counter for an exercise."""
    config = EXERCISE_CONFIG[name]
    lm_a, lm_b, lm_c = config["landmarks"]
    counter = RepCounter(config["down_threshold"], config["up_threshold"])
    return config, lm_a, lm_b, lm_c, counter


def main():
    args = parse_args()
    exercise = args.exercise if args.exercise else select_exercise()
    config, lm_a, lm_b, lm_c, counter = switch_exercise(exercise)

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
        current_angle = None

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
            current_angle = calculate_angle_3d(a, b, c)

            # Count reps unless visibility is too low
            if can_count:
                counter.update(current_angle)

            # --- Draw skeleton ---
            drawing_utils.draw_landmarks(
                frame, landmarks, pose_connections
            )

            # --- Draw angle arc near joint ---
            draw_angle_arc(frame, current_angle, config, landmarks[lm_b], w, h)
        else:
            warnings.append(("No pose detected", RED))

        # --- HUD ---
        draw_hud(frame, exercise, counter.count, counter.stage,
                 current_angle, config, warnings)

        # --- Exercise selector bar ---
        draw_exercise_bar(frame, exercise)

        cv2.imshow("RepCount", frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counter.reset()
            print("Rep counter reset.")
        else:
            idx = -1
            if ord("1") <= key <= ord("9"):
                idx = key - ord("1")
            elif key == ord("0"):
                idx = 9
            elif ord("a") <= key <= ord("z"):
                idx = 10 + (key - ord("a"))
                
            if 0 <= idx < len(EXERCISE_LIST):
                exercise = EXERCISE_LIST[idx]
                config, lm_a, lm_b, lm_c, counter = switch_exercise(exercise)
                print(f"Switched to: {exercise.replace('_', ' ').title()}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
