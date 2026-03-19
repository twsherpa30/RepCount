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
repcount/
├── main.py           # Entry point — webcam loop, UI, and orchestration
├── rep_counter.py    # UP/DOWN state machine with angle smoothing
├── utils.py          # 2D and 3D joint angle calculation helpers
├── validator.py      # Position and visibility checks
└── requirements.txt  # mediapipe, opencv-python, numpy
```

---

## Supported Exercises

| Exercise | Joint Tracked | Down Threshold | Up Threshold |
|----------|---------------|----------------|--------------|
| `squat` | Knee (hip→knee→ankle) | < 90° | > 160° |
| `pushup` | Elbow (shoulder→elbow→wrist) | < 90° | > 160° |
| `bicep_curl` | Elbow (shoulder→elbow→wrist) | < 40° | > 140° |
| `shoulder_press` | Shoulder (elbow→shoulder→hip) | < 90° | > 160° |
| `deadlift` | Hip (shoulder→hip→knee) | < 90° | > 160° |
| `lunge` | Knee (hip→knee→ankle) | < 90° | > 160° |
| `lateral_raise` | Shoulder (hip→shoulder→elbow) | < 30° | > 80° |

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
- **Q** — quit
- **R** — reset rep count

---

## Phone Camera Setup

### Option 1: Continuity Camera (macOS Ventura+)
1. Ensure both devices are on the same Apple ID with Wi-Fi and Bluetooth enabled
2. Lock your iPhone and bring it near your Mac
3. Run with `--camera 1` or `--camera 2`


---

## How It Works

- **3D Angle Calculation:** Uses MediaPipe's x, y, z coordinates with dot-product formula for accurate joint angles regardless of camera orientation
- **Angle Smoothing:** Rolling average over 5 frames reduces jitter and prevents false counts
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

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Full body should be in frame | Cropped joints break angle calculation |
| Lighting | Poor light reduces landmark visibility |
| Clothing | Baggy clothing obscures joint positions |
| Side-on is best | Side view gives the most accurate 2D angle cues |
