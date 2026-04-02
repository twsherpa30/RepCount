"""Rep counting state machine for exercise tracking."""

import time
from collections import deque


class RepCounter:
    """Tracks exercise reps using an UP/DOWN state machine based on joint angle.

    Features:
        - Grace period on start/reset to avoid false counts while positioning.
        - Median smoothing to filter outlier spikes from bad pose frames.
        - Hysteresis band to prevent rapid state toggling near thresholds.
        - Time-based debounce to prevent double-counting from noisy data.

    Args:
        down_threshold: Angle (degrees) below which the exercise is in the
                        "down" position.
        up_threshold:   Angle (degrees) above which the exercise is in the
                        "up" position.
        smooth_window:  Number of frames to use for median smoothing.
        hysteresis:     Extra degrees beyond threshold required to trigger
                        a state change (prevents jitter near boundaries).
        debounce_sec:   Minimum seconds between consecutive rep counts.
        grace_sec:      Seconds after start/reset during which reps are NOT
                        counted (allows user to get into position).
    """

    def __init__(self, down_threshold=90, up_threshold=160,
                 smooth_window=7, hysteresis=3, debounce_sec=0.4,
                 grace_sec=3.0):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.hysteresis = hysteresis
        self.debounce_sec = debounce_sec
        self.grace_sec = grace_sec
        self.count = 0
        self.stage = None  # None -> "down" -> "up" (one rep)
        self._buffer = deque(maxlen=smooth_window)
        self._last_rep_time = 0.0
        self._start_time = time.monotonic()

    @property
    def in_grace_period(self):
        """True while the startup grace period is active."""
        return time.monotonic() - self._start_time < self.grace_sec

    @property
    def grace_remaining(self):
        """Seconds remaining in the grace period (0 if expired)."""
        remaining = self.grace_sec - (time.monotonic() - self._start_time)
        return max(0.0, remaining)

    def _median(self):
        """Return the median of the angle buffer."""
        vals = sorted(self._buffer)
        n = len(vals)
        mid = n // 2
        if n % 2 == 0:
            return (vals[mid - 1] + vals[mid]) / 2.0
        return vals[mid]

    def update(self, angle):
        """Update the state machine with a smoothed angle reading.

        Applies a median filter over the last *smooth_window* frames,
        then checks thresholds with hysteresis and debounce.

        During the grace period, angles are buffered and state is tracked
        but NO reps are counted.  This prevents false counts while the
        user is still getting into position.

        Returns:
            tuple: (count, stage) — current rep count and stage string.
        """
        self._buffer.append(angle)
        if len(self._buffer) < 3:
            return self.count, self.stage

        smoothed = self._median()

        # Hysteresis: require crossing threshold by an extra margin
        down_trigger = self.down_threshold - self.hysteresis
        up_trigger = self.up_threshold + self.hysteresis

        if smoothed < down_trigger:
            self.stage = "down"

        if smoothed > up_trigger and self.stage == "down":
            now = time.monotonic()
            if now - self._start_time >= self.grace_sec:
                # Past grace period — count normally with debounce
                if now - self._last_rep_time >= self.debounce_sec:
                    self.stage = "up"
                    self.count += 1
                    self._last_rep_time = now
            else:
                # During grace period — transition state but don't count
                self.stage = "up"

        return self.count, self.stage

    def reset(self):
        """Reset the rep counter to its initial state."""
        self.count = 0
        self.stage = None
        self._buffer.clear()
        self._last_rep_time = 0.0
        self._start_time = time.monotonic()
