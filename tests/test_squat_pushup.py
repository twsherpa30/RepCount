"""Quick threshold sanity checks for home workout exercises."""

from src.rep_counter import RepCounter


def test_squat_counts():
    rc = RepCounter(down_threshold=95, up_threshold=155, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(80)
    for _ in range(5):
        rc.update(170)
    assert rc.count == 1, f"Expected 1 squat rep, got {rc.count}"


def test_pushup_counts():
    rc = RepCounter(down_threshold=100, up_threshold=135, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(85)
    for _ in range(5):
        rc.update(150)
    assert rc.count == 1, f"Expected 1 pushup rep, got {rc.count}"


def test_sit_up_counts():
    rc = RepCounter(down_threshold=95, up_threshold=140, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(75)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(155)
    assert rc.count == 1, f"Expected 1 sit-up rep, got {rc.count}"


def test_jumping_jack_counts():
    rc = RepCounter(down_threshold=35, up_threshold=70, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(15)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(90)
    assert rc.count == 1, f"Expected 1 jumping jack rep, got {rc.count}"


def test_glute_bridge_counts():
    rc = RepCounter(down_threshold=125, up_threshold=150, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(105)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(165)
    assert rc.count == 1, f"Expected 1 glute bridge rep, got {rc.count}"


def test_leg_raise_counts():
    rc = RepCounter(down_threshold=110, up_threshold=150, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(90)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(165)
    assert rc.count == 1, f"Expected 1 leg raise rep, got {rc.count}"


def test_arm_raise_counts():
    rc = RepCounter(down_threshold=30, up_threshold=90, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(15)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(100)
    assert rc.count == 1, f"Expected 1 arm raise rep, got {rc.count}"


def test_high_knees_counts():
    rc = RepCounter(down_threshold=115, up_threshold=145, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(95)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(160)
    assert rc.count == 1, f"Expected 1 high knee rep, got {rc.count}"


def test_tricep_dip_counts():
    rc = RepCounter(down_threshold=95, up_threshold=135, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(80)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(150)
    assert rc.count == 1, f"Expected 1 tricep dip rep, got {rc.count}"


def test_lunge_counts():
    rc = RepCounter(down_threshold=95, up_threshold=155, hysteresis=2, debounce_sec=0, grace_sec=0)
    for _ in range(5):
        rc.update(80)
    assert rc.stage == "down"
    for _ in range(5):
        rc.update(170)
    assert rc.count == 1, f"Expected 1 lunge rep, got {rc.count}"


def test_no_false_count_near_threshold():
    """Hovering near a threshold should NOT trigger a rep."""
    rc = RepCounter(down_threshold=95, up_threshold=155, hysteresis=2, debounce_sec=0, grace_sec=0)
    # Hover around the down threshold (but never cross with hysteresis)
    for angle in [96, 94, 96, 94, 96]:
        rc.update(angle)
    # Hover around the up threshold
    for angle in [154, 156, 154, 156, 154]:
        rc.update(angle)
    assert rc.count == 0, f"Expected 0 reps from threshold hovering, got {rc.count}"


def test_multiple_reps():
    """Verify consecutive reps are counted correctly."""
    rc = RepCounter(down_threshold=95, up_threshold=155, hysteresis=2, debounce_sec=0, grace_sec=0)
    for rep in range(3):
        for _ in range(5):
            rc.update(80)
        for _ in range(5):
            rc.update(170)
    assert rc.count == 3, f"Expected 3 reps, got {rc.count}"
