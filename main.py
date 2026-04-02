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
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

from src.config import (
    EXERCISE_CONFIG, EXERCISE_LIST, MODEL_PATH, ORANGE, RED,
)
from src.camera import open_camera
from src.drawing import draw_hud, draw_exercise_bar, draw_angle_arc
from src.rep_counter import RepCounter
from src.utils import calculate_angle_3d
from src.validator import check_visibility, is_side_on, is_body_in_frame


# ---------------------------------------------------------------------------
# CLI & menu helpers
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    exercise = args.exercise if args.exercise else select_exercise()
    config, lm_a, lm_b, lm_c, counter = switch_exercise(exercise)

    cap = open_camera(args.camera, args.camera_url)

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
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

        # Ensure monotonically increasing timestamps for VIDEO mode
        current_time_ms = int(time.time() * 1000)
        if not hasattr(landmarker, 'last_timestamp_ms'):
            landmarker.last_timestamp_ms = 0

        timestamp_ms = max(current_time_ms, landmarker.last_timestamp_ms + 1)
        landmarker.last_timestamp_ms = timestamp_ms

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
            drawing_utils.draw_landmarks(frame, landmarks, pose_connections)

            # --- Draw angle arc near joint ---
            draw_angle_arc(frame, current_angle, config, landmarks[lm_b], w, h)
        else:
            warnings.append(("No pose detected", RED))

        # --- HUD ---
        draw_hud(frame, exercise, counter.count, counter.stage,
                 current_angle, config, warnings,
                 grace_remaining=counter.grace_remaining)

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
