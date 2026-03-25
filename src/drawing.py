"""HUD and overlay drawing functions for RepCount."""

import cv2
import numpy as np

from src.config import (
    EXERCISE_LIST, EXERCISE_CONFIG,
    BG_ALPHA, ACCENT, ACCENT_DIM, ORANGE, RED, WHITE, GRAY, DARK,
    STAGE_UP_COLOR, STAGE_DOWN_COLOR,
    HUD_FONT, HUD_FONT_ITALIC,
)


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
