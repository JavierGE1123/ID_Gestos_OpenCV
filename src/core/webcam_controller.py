import cv2

class WebcamController:

    def __init__(self, source=1):
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise Exception("Could not open webcam")

    def get_frame(self):
        ret, frame = self.capture.read()
        return ret, frame

    def release(self):
        self.capture.release()