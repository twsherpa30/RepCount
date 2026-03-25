"""Camera initialisation and threaded capture for RepCount."""

import threading

import cv2


class ThreadedCamera:
    """Reads frames from cv2.VideoCapture in a separate thread to prevent buffer lag."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.running = False
            return

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.ret = ret
            self.frame = frame

        self.running = False

    def read(self):
        return self.ret, self.frame

    def isOpened(self):
        return self.running and self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)


def open_camera(camera_index=None, camera_url=None):
    """Open a video capture source.

    Priority: camera_url > camera_index > auto-detect (indices 0-2).
    """
    if camera_url:
        cap = ThreadedCamera(camera_url)
        if cap.isOpened():
            print(f"Camera opened on URL: {camera_url}")
            return cap
        raise RuntimeError(f"Cannot open camera at URL: {camera_url}")

    if camera_index is not None:
        cap = ThreadedCamera(camera_index)
        if cap.isOpened():
            print(f"Camera opened on index {camera_index}")
            return cap
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    for idx in (0, 1, 2):
        cap = ThreadedCamera(idx)
        if cap.isOpened():
            print(f"Camera opened on index {idx}")
            return cap
    raise RuntimeError("No camera found on indices 0, 1, or 2.")
