from src.rep_counter import RepCounter

def test_squat_with_current_thresholds():
    # 90 down, 160 up
    rc = RepCounter(down_threshold=90, up_threshold=160, hysteresis=3, debounce_sec=0)
    # realistic parallel squat (around 95 deg)
    for _ in range(5):
        rc.update(95)
    for _ in range(5):
        rc.update(155)
    print("Squat (current 90/160), angle 95 -> 155:", rc.count)

def test_squat_with_new_thresholds():
    # 110 down, 150 up
    rc = RepCounter(down_threshold=110, up_threshold=150, hysteresis=3, debounce_sec=0)
    # realistic soft squat (105 deg)
    for _ in range(5):
        rc.update(105)
    for _ in range(5):
        rc.update(155)
    print("Squat (new 110/150), angle 105 -> 155:", rc.count)

test_squat_with_current_thresholds()
test_squat_with_new_thresholds()
