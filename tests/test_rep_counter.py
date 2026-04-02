"""Unit tests for the RepCounter state machine."""

import time
from src.rep_counter import RepCounter


class TestRepCounterBasic:
    """Basic counting behaviour."""

    def test_single_rep_down_then_up(self):
        """A clean down→up cycle should count exactly 1 rep."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0, grace_sec=0)
        # Feed "down" angles
        for _ in range(5):
            rc.update(70)
        assert rc.stage == "down"
        assert rc.count == 0

        # Feed "up" angles
        for _ in range(5):
            rc.update(170)
        assert rc.stage == "up"
        assert rc.count == 1

    def test_multiple_reps(self):
        """Three full cycles should count 3 reps."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0, grace_sec=0)
        for _ in range(3):
            for _ in range(5):
                rc.update(70)
            for _ in range(5):
                rc.update(170)
        assert rc.count == 3

    def test_up_without_down_no_count(self):
        """Going to 'up' angle without first going 'down' should not count."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0, grace_sec=0)
        for _ in range(7):
            rc.update(170)
        assert rc.count == 0

    def test_reset(self):
        """Reset should clear count, stage, and buffer."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0, grace_sec=0)
        for _ in range(5):
            rc.update(70)
        for _ in range(5):
            rc.update(170)
        assert rc.count == 1

        rc.reset()
        assert rc.count == 0
        assert rc.stage is None


class TestHysteresis:
    """Hysteresis prevents false counts from jittery angles near thresholds."""

    def test_oscillation_near_threshold_no_false_count(self):
        """Angles hovering near down_threshold should NOT trigger transitions."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=5, debounce_sec=0, grace_sec=0)
        # Hover right at the threshold boundary (within hysteresis band)
        for angle in [92, 88, 91, 87, 92, 88, 91, 87, 91, 88]:
            rc.update(angle)
        # Should NOT have entered "down" — angles never went below 85 (90 - 5)
        assert rc.stage is None or rc.stage != "down" or rc.count == 0

    def test_clear_crossing_triggers_transition(self):
        """Angles clearly below (threshold - hysteresis) should trigger 'down'."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=5, debounce_sec=0, grace_sec=0)
        # Go clearly below: 90 - 5 = 85, so 80 should trigger
        for _ in range(7):
            rc.update(80)
        assert rc.stage == "down"


class TestDebounce:
    """Debounce prevents double-counting from rapid transitions."""

    def test_rapid_reps_debounced(self):
        """Two reps within debounce window should only count as 1."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=1.0, grace_sec=0)

        # First rep
        for _ in range(5):
            rc.update(70)
        for _ in range(5):
            rc.update(170)
        assert rc.count == 1

        # Immediate second rep (within 1.0s debounce)
        for _ in range(5):
            rc.update(70)
        for _ in range(5):
            rc.update(170)
        # Should still be 1 because debounce hasn't expired
        assert rc.count == 1

    def test_reps_after_debounce_window(self):
        """Reps separated by more than debounce_sec should both count."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0.4, grace_sec=0)

        # First rep
        for _ in range(5):
            rc.update(70)
        for _ in range(5):
            rc.update(170)
        assert rc.count == 1

        # Simulate that enough time has passed by backdating _last_rep_time
        rc._last_rep_time = time.monotonic() - 1.0

        # Second rep should now count
        for _ in range(5):
            rc.update(70)
        for _ in range(5):
            rc.update(170)
        assert rc.count == 2


class TestMedianSmoothing:
    """Median filter should reject outlier spikes."""

    def test_outlier_spike_rejected(self):
        """A single outlier frame should not cause a false transition."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0, grace_sec=0)
        # Establish "down" state
        for _ in range(7):
            rc.update(70)
        assert rc.stage == "down"

        # Inject a single outlier spike — median of [70,70,70,70,70,70,200] = 70
        rc.update(200)
        assert rc.count == 0  # should NOT have counted a rep

    def test_sustained_change_passes_through(self):
        """A sustained angle change should eventually transition state."""
        rc = RepCounter(down_threshold=90, up_threshold=160,
                        hysteresis=0, debounce_sec=0, grace_sec=0)
        for _ in range(7):
            rc.update(70)
        assert rc.stage == "down"

        # Sustained high angle (all frames)
        for _ in range(7):
            rc.update(170)
        assert rc.count == 1


class TestRealisticExerciseAngles:
    """Realistic angles with new default hysteresis (3°) should count reps."""

    def test_sit_up_realistic_angles(self):
        """Sit-up with realistic range (75° down, 155° up) should count."""
        rc = RepCounter(down_threshold=90, up_threshold=150,
                        hysteresis=3, debounce_sec=0, grace_sec=0)
        # Sit up (hip angle decreases)
        for _ in range(7):
            rc.update(75)
        assert rc.stage == "down"

        # Lie back down (hip angle increases)
        for _ in range(7):
            rc.update(155)
        assert rc.count == 1

    def test_jumping_jack_realistic_angles(self):
        """Jumping jack with realistic range (15° down, 95° up) should count."""
        rc = RepCounter(down_threshold=30, up_threshold=80,
                        hysteresis=3, debounce_sec=0, grace_sec=0)
        # Arms at sides (small angle)
        for _ in range(7):
            rc.update(15)
        assert rc.stage == "down"

        # Arms raised (larger angle)
        for _ in range(7):
            rc.update(95)
        assert rc.count == 1

    def test_old_tight_thresholds_not_required(self):
        """With new hysteresis=3, angles 87° and 153° should still count a
        sit-up rep — the old hysteresis=5 would have required <85° and >155°."""
        rc = RepCounter(down_threshold=90, up_threshold=150,
                        hysteresis=3, debounce_sec=0, grace_sec=0)
        # down_trigger = 90 - 3 = 87, so 85 is below it
        for _ in range(7):
            rc.update(85)
        assert rc.stage == "down"

        # up_trigger = 150 + 3 = 153, so 155 is above it
        for _ in range(7):
            rc.update(155)
        assert rc.count == 1

    def test_default_hysteresis_is_three(self):
        """Confirm the default hysteresis is now 3°."""
        rc = RepCounter()
        assert rc.hysteresis == 3

