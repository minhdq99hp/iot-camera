import cv2
from contextlib import contextmanager
import os


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


class FrameGenerator:
    def __init__(self, mode, input_path):
        self.mode = mode # 0: image 1: folder of images 2: video 3:rtsp stream
        self.path = input_path

    def yield_frame(self):
        if self.mode == 0:
            return self.yield_frame_from_image()
        elif self.mode == 1:
            return self.yield_frame_from_image_dir()
        elif self.mode == 2:
            return self.yield_frame_from_video()
        elif self.mode == 3:
            return self.yield_frame_from_rtsp()

    def yield_frame_from_image(self):
        yield cv2.imread(self.path)


    def yield_frame_from_video(self):
        with video_capture(self.path) as cap:
            while True:
                ret, img = cap.read()
                if ret:
                    # raise RuntimeError("Failed to capture image")
                    yield img
                else:
                    break

    def yield_frame_from_rtsp(self):
        with video_capture(self.path) as cap:
            while True:
                ret, img = cap.read()
                if ret:
                    # raise RuntimeError("Failed to capture image")
                    yield img
                else:
                    break

    def yield_frame_from_image_dir(self):
        for file in os.listdir(self.path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = self.path + file if self.path.endswith('/') else self.path + '/' + file
                yield cv2.imread(file_path)




