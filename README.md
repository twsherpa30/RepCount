# RepCount — Exercise Rep Counter Using Pose Estimation

A real-time webcam-based rep counter for gym exercises using MediaPipe Pose estimation and 3D joint angle logic.

---

## Tech Stack

- **Language:** Python 3.9+
- **Pose Detection:** MediaPipe Pose (3D landmarks)
- **Camera/Display:** OpenCV
- **Math:** NumPy
- **Camera:** Webcam, iPhone via Continuity Camera, or IP camera stream

---

## Project Structure

```
RepCount/
├── main.py                  # Entry point — CLI, exercise menu, main loop
├── requirements.txt         # mediapipe, opencv-python, numpy
├── README.md
├── models/
│   └── pose_landmarker.task # MediaPipe pose landmarker model
├── src/
│   ├── __init__.py
│   ├── config.py            # Exercise definitions, colors, constants
│   ├── camera.py            # Threaded camera capture + auto-detect
│   ├── drawing.py           # HUD overlay, exercise bar, angle arc
│   ├── rep_counter.py       # UP/DOWN state machine with smoothing
│   ├── utils.py             # 2D and 3D joint angle calculation
│   └── validator.py         # Visibility and position checks
└── tests/
    ├── __init__.py
    ├── test_rep_counter.py  # Unit tests for rep counting logic
    ├── test_cam.py          # Camera test script
    └── test_lag.py          # Lag test script
```

---

## Supported Exercises

| Exercise | Joint Tracked | Down Threshold | Up Threshold |
|----------|---------------|----------------|--------------|
| `squat` | Knee (hip→knee→ankle) | < 90° | > 160° |
| `pushup` | Elbow (shoulder→elbow→wrist) | < 100° | > 140° |
| `lunge` | Knee (hip→knee→ankle) | < 90° | > 160° |
| `sit_up` | Hip (shoulder→hip→knee) | < 90° | > 150° |
| `glute_bridge` | Hip (shoulder→hip→knee) | < 120° | > 155° |
| `tricep_dip` | Elbow (shoulder→elbow→wrist) | < 90° | > 140° |
| `jumping_jack` | Shoulder (hip→shoulder→elbow) | < 30° | > 80° |
| `leg_raise` | Hip (shoulder→hip→knee) | < 110° | > 160° |
| `high_knees` | Hip (shoulder→hip→knee) | < 110° | > 155° |
| `arm_raise` | Shoulder (hip→shoulder→elbow) | < 25° | > 120° |

---

## Usage

```bash
# Default — squat with built-in webcam
python main.py

# Choose exercise
python main.py --exercise bicep_curl

# Use a specific camera index (e.g. iPhone via Continuity Camera)
python main.py --camera 1

# Use iPhone as IP camera (e.g. via DroidCam, IPCamera Lite, etc.)
python main.py --camera-url http://192.168.1.5:4747/video

# Combine options
python main.py --exercise shoulder_press --camera-url http://192.168.1.5:4747/video
```

### Controls
- **1–9, 0** — switch exercise on the fly (see exercise list below)
- **R** — reset rep count
- **Q** — quit

---

## Phone Camera Setup

### Option 1: Continuity Camera (macOS Ventura+)
1. Ensure both devices are on the same Apple ID with Wi-Fi and Bluetooth enabled
2. Lock your iPhone and bring it near your Mac
3. Run with `--camera 1` or `--camera 2`


---

## How It Works

- **3D Angle Calculation:** Uses MediaPipe's x, y, z coordinates with dot-product formula for accurate joint angles regardless of camera orientation
- **Median Smoothing:** Rolling median over 7 frames rejects outlier spikes from bad pose frames
- **Hysteresis:** 5° band around thresholds prevents jitter-induced state toggling
- **Debounce:** 0.4s minimum between reps prevents double-counting
- **State Machine:** Reps are counted on a down→up transition (angle drops below threshold, then rises above)
- **Validator:** Checks landmark visibility (blocks counting if unreliable), body positioning (advisory tips)

---

## Installation

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install mediapipe opencv-python numpy
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Full body should be in frame | Cropped joints break angle calculation |
| Lighting | Poor light reduces landmark visibility |
| Clothing | Baggy clothing obscures joint positions |
| Side-on is best | Side view gives the most accurate 2D angle cues |
