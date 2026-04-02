"""Quick threshold sanity checks for home workout exercises."""

from src.rep_counter import RepCounter


def test_squat_counts():
    rc = RepCounter(down_threshold=90, up_threshold=160, hysteresis=3, debounce_sec=0, grace_sec=0)
    for _ in range(7):
        rc.update(80)
    for _ in range(7):
        rc.update(170)
    assert rc.count == 1, f"Expected 1 squat rep, got {rc.count}"


def test_pushup_counts():
    rc = RepCounter(down_threshold=100, up_threshold=140, hysteresis=3, debounce_sec=0, grace_sec=0)
    for _ in range(7):
        rc.update(85)
    for _ in range(7):
        rc.update(150)
    assert rc.count == 1, f"Expected 1 pushup rep, got {rc.count}"


def test_sit_up_counts():
    rc = RepCounter(down_threshold=90, up_threshold=150, hysteresis=3, debounce_sec=0, grace_sec=0)
    # Sit up (hip angle decreases)
    for _ in range(7):
        rc.update(75)
    assert rc.stage == "down"
    # Lie back (hip angle increases)
    for _ in range(7):
        rc.update(160)
    assert rc.count == 1, f"Expected 1 sit-up rep, got {rc.count}"


def test_jumping_jack_counts():
    rc = RepCounter(down_threshold=30, up_threshold=80, hysteresis=3, debounce_sec=0, grace_sec=0)
    # Arms at sides (small shoulder angle)
    for _ in range(7):
        rc.update(15)
    assert rc.stage == "down"
    # Arms raised (large shoulder angle)
    for _ in range(7):
        rc.update(100)
    assert rc.count == 1, f"Expected 1 jumping jack rep, got {rc.count}"


def test_glute_bridge_counts():
    rc = RepCounter(down_threshold=120, up_threshold=155, hysteresis=3, debounce_sec=0, grace_sec=0)
    # Hips on floor (hip angle small)
    for _ in range(7):
        rc.update(105)
    assert rc.stage == "down"
    # Hips raised (hip angle large)
    for _ in range(7):
        rc.update(165)
    assert rc.count == 1, f"Expected 1 glute bridge rep, got {rc.count}"


def test_leg_raise_counts():
    rc = RepCounter(down_threshold=110, up_threshold=160, hysteresis=3, debounce_sec=0, grace_sec=0)
    # Leg raised
    for _ in range(7):
        rc.update(90)
    assert rc.stage == "down"
    # Leg lowered
    for _ in range(7):
        rc.update(170)
    assert rc.count == 1, f"Expected 1 leg raise rep, got {rc.count}"
