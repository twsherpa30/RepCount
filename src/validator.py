"""Position and visibility validation for pose estimation."""

# Landmark indices used per exercise
EXERCISE_LANDMARKS = {
    "squat": [23, 25, 27],          # hip, knee, ankle
    "pushup": [11, 13, 15],         # shoulder, elbow, wrist
    "bicep_curl": [11, 13, 15],     # shoulder, elbow, wrist
    "shoulder_press": [13, 11, 23], # elbow, shoulder, hip
    "deadlift": [11, 23, 25],       # shoulder, hip, knee
    "lunge": [23, 25, 27],          # hip, knee, ankle
    "lateral_raise": [23, 11, 13],  # hip, shoulder, elbow
    "bench_press": [11, 13, 15],    # shoulder, elbow, wrist
    "leg_press": [23, 25, 27],      # hip, knee, ankle
    "pullup": [11, 13, 15],         # shoulder, elbow, wrist
}


def check_visibility(landmarks, exercise):
    """Return a list of landmark indices with low visibility.

    Counting is only paused when the *majority* of the exercise's
    landmarks are below the visibility threshold, so a single
    partially-occluded joint won't freeze the counter.

    Args:
        landmarks: MediaPipe pose landmarks (list-like, indexed by landmark id).
        exercise:  Exercise name string.

    Returns:
        list[int]: Indices of landmarks whose visibility is below the
                   threshold.  The list is only returned (non-empty) when
                   at least 2 out of 3 landmarks are below the threshold;
                   otherwise an empty list is returned so counting continues.
    """
    indices = EXERCISE_LANDMARKS.get(exercise, [])
    low_vis = []
    for idx in indices:
        if landmarks[idx].visibility < 0.3:
            low_vis.append(idx)
    # Only signal "low visibility" when the majority of joints are invisible
    if len(low_vis) < 2:
        return []
    return low_vis


def is_side_on(landmarks, threshold=0.25):
    """Check whether the user is oriented side-on to the camera.

    Compares the x-coordinates of the left shoulder (11) and right
    shoulder (12). If the difference is small, the user is side-on.

    Returns:
        bool: True if the user is side-on (good positioning).
    """
    left_x = landmarks[11].x
    right_x = landmarks[12].x
    return abs(left_x - right_x) < threshold


# Key body landmarks (hips, shoulders, knees, ankles) — skip extremities
# like fingertips and toes that are often off-screen.
KEY_BODY_INDICES = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles


def is_body_in_frame(landmarks, margin=0.05):
    """Check whether key body landmarks are within the visible frame.

    Only checks shoulders, hips, knees, and ankles — not fingertips,
    toes, or face landmarks which are often partially off-screen.

    Returns:
        bool: True if key landmarks are comfortably within the frame.
    """
    for idx in KEY_BODY_INDICES:
        lm = landmarks[idx]
        if not (margin <= lm.x <= 1 - margin and margin <= lm.y <= 1 - margin):
            return False
    return True
