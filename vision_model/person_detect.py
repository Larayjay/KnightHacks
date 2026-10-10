"""
Basic real-time person detection with YOLO (Ultralytics).
Press 'q' to quit.

Setup (once):
    pip install ultralytics opencv-python
Run:
    python person_detect.py
"""
import cv2
from ultralytics import YOLO

# Small, fast model; downloads automatically on first run (~6 MB)
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(1)  # 0 = default webcam; or a video file path like "video.mp4"
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Try another index (1, 2) or a video file path.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # classes=[0] -> only detect "person" (COCO class 0)
    results = model(frame, classes=[0], conf=0.5, verbose=False)

    people = len(results[0].boxes)
    annotated = results[0].plot()  # draws boxes + confidence
    cv2.putText(annotated, f"People: {people}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Person Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()