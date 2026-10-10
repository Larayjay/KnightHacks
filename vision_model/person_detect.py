"""
Person detection (YOLO) + fatigue monitoring with PERCLOS (MediaPipe face landmarks).

Video source: a Raspberry Pi camera streamed over TCP (rpicam-vid), or a local webcam.

PERCLOS = percentage of time, over a rolling window, that the eyes are (mostly) closed.
Eye closure is measured with the Eye Aspect Ratio (EAR) from face landmarks, compared
to the person's own open-eye baseline (calibrated for a few seconds at startup).

Setup (once, inside your venv):
    pip install ultralytics opencv-python mediapipe
On the Pi (keeps the stream available, restarts after each disconnect):
    while true; do rpicam-vid -t 0 --inline --listen --width 1280 --height 720 --framerate 30 -o tcp://0.0.0.0:8888; done
Run (on the Mac):
    python fatigue_monitor.py
Keys:
    q = quit    c = recalibrate (keep eyes normally open for a few seconds)
"""
import os

# Low-latency FFmpeg options for the network stream. Must be set BEFORE importing cv2.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ---------------- Settings ----------------
# Raspberry Pi stream. Use the Pi's hotspot IP (172.20.10.2) or its Tailscale IP (100.111.60.44).
# To go back to the Mac webcam, set this to an int, e.g. CAM_SOURCE = 1
CAM_SOURCE = "tcp://172.20.10.2:8888"

PERCLOS_WINDOW_S = 60     # rolling window; 60 s is the standard-ish choice
CALIBRATION_S = 3         # seconds of open-eye baseline at startup
CLOSED_RATIO = 0.45       # "closed" when EAR < baseline * this (~80% lid closure, i.e. P80)
DROWSY_PERCLOS = 0.15     # >=15% of the window closed -> drowsy
SEVERE_PERCLOS = 0.30     # >=30% -> very drowsy
LONG_CLOSURE_S = 2.0      # eyes continuously closed this long -> alert

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe face mesh indices: p1(outer corner), p2, p3 (upper lid), p4(inner corner), p5, p6 (lower lid)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

GREEN, ORANGE, RED, GRAY, YELLOW = (0, 200, 0), (0, 165, 255), (0, 0, 255), (160, 160, 160), (0, 255, 255)


class LatestFrameReader:
    """Reads frames on a background thread and keeps only the newest one.

    Without this, if YOLO + MediaPipe run slower than the stream's frame rate,
    frames pile up in the buffer and the video falls further and further behind.
    Also reconnects automatically if the stream drops (e.g. the Pi restarts rpicam-vid).
    """

    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        if isinstance(self.source, int):
            return cv2.VideoCapture(self.source)
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def _run(self):
        while self.running:
            cap = self._open()
            if not cap.isOpened():
                print(f"Waiting for video at {self.source} ... (is rpicam-vid running on the Pi?)")
                cap.release()
                time.sleep(1)
                continue
            print(f"Connected to {self.source}")
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
                    self.seq += 1
            cap.release()
            if self.running:
                print("Stream lost, reconnecting...")
                time.sleep(1)

    def read(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.seq

    def stop(self):
        self.running = False


def ensure_model():
    if MODEL_PATH.exists():
        return
    print("Downloading face landmark model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise SystemExit(
            f"Download failed ({e}).\nDownload it manually in this folder with:\n"
            f"  curl -L -o face_landmarker.task {MODEL_URL}"
        )


def eye_aspect_ratio(p):
    d = lambda a, b: np.linalg.norm(np.subtract(a, b))
    return (d(p[1], p[5]) + d(p[2], p[4])) / (2.0 * d(p[0], p[3]) + 1e-6)


def put(img, text, row, color):
    cv2.putText(img, text, (10, 30 + 32 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 30 + 32 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    ensure_model()
    yolo = YOLO("yolo11n.pt")
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )

    reader = LatestFrameReader(CAM_SOURCE)

    history = deque()          # (timestamp, eyes_closed)
    calib, calib_start, baseline = [], None, None
    closed_since, last_ts = None, 0
    last_seq = 0

    while True:
        frame, seq = reader.read()
        if frame is None or seq == last_seq:
            # No new frame yet: keep the window responsive and wait briefly.
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            time.sleep(0.005)
            continue
        last_seq = seq

        now = time.monotonic()
        h, w = frame.shape[:2]

        # --- People (YOLO) ---
        det = yolo(frame, classes=[0], conf=0.5, verbose=False)[0]
        out = det.plot()
        people = len(det.boxes)

        # --- Face / eyes (MediaPipe) ---
        ts = max(int(now * 1000), last_ts + 1)  # must strictly increase
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        ear = None
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            pts = lambda idx: [(lm[i].x * w, lm[i].y * h) for i in idx]
            left, right = pts(LEFT_EYE), pts(RIGHT_EYE)
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2
            for x, y in left + right:
                cv2.circle(out, (int(x), int(y)), 2, YELLOW, -1)

            if baseline is None:
                calib_start = calib_start or now
                calib.append(ear)
                if now - calib_start >= CALIBRATION_S:
                    baseline = float(np.percentile(calib, 75))  # robust to blinks during calibration
            else:
                closed = ear < baseline * CLOSED_RATIO
                history.append((now, closed))
                closed_since = (closed_since or now) if closed else None
        else:
            closed_since = None

        while history and now - history[0][0] > PERCLOS_WINDOW_S:
            history.popleft()

        # --- Status ---
        perclos = sum(c for _, c in history) / len(history) if history else 0.0
        window_filled = (now - history[0][0]) if history else 0.0

        if ear is None:
            status, color = "No face visible", GRAY
        elif baseline is None:
            left_s = max(0, CALIBRATION_S - (now - calib_start))
            status, color = f"Calibrating - eyes open ({left_s:.0f}s)", YELLOW
        elif closed_since and now - closed_since >= LONG_CLOSURE_S:
            status, color = f"EYES CLOSED {now - closed_since:.0f}s", RED
        elif perclos >= SEVERE_PERCLOS:
            status, color = "VERY DROWSY", RED
        elif perclos >= DROWSY_PERCLOS:
            status, color = "DROWSY", ORANGE
        else:
            status, color = "Alert", GREEN

        put(out, f"People: {people}", 0, GREEN)
        put(out, f"EAR: {ear:.3f}" + (f" (base {baseline:.3f})" if baseline else "") if ear else "EAR: -", 1, (255, 255, 255))
        warm = "" if window_filled >= PERCLOS_WINDOW_S * 0.9 else f"  (warming up {window_filled:.0f}/{PERCLOS_WINDOW_S}s)"
        put(out, f"PERCLOS: {perclos * 100:.0f}%{warm}", 2, (255, 255, 255))
        put(out, status, 3, color)

        cv2.imshow("Person + Fatigue Monitor", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            calib, calib_start, baseline, closed_since = [], None, None, None
            history.clear()

    reader.stop()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()