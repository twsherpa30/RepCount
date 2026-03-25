import cv2
import time

cap = cv2.VideoCapture(0)
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("test", frame)
    # simulate 100ms processing delay
    time.sleep(0.1)
    if cv2.waitKey(1) == ord('q'): break
