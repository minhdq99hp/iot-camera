import cv2
from contextlib import contextmanager
import os
from enum import Enum


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


class StreamMode(Enum):
    IMAGE = 0
    IMAGE_DIR = 1
    VIDEO = 2
    RTSP = 3


class FrameGenerator:
    def __init__(self, mode, input_path):
        self.mode = mode
        self.path = input_path

        # SETUP IF IT IS VIDEO
        if self.mode == StreamMode.VIDEO:
            vid = cv2.VideoCapture(input_path)
            self.vid_fps = vid.get(cv2.CAP_PROP_FPS)
            self.vid_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            if not vid.isOpened():
                raise Exception(f"OpenCV can't open {input_path}")
            vid.release()

    def yield_frame(self):
        if self.mode == StreamMode.IMAGE:
            return self.yield_frame_from_image()
        elif self.mode == StreamMode.IMAGE_DIR:
            return self.yield_frame_from_image_dir()
        elif self.mode == StreamMode.VIDEO:
            return self.yield_frame_from_video()
        elif self.mode == StreamMode.RTSP:
            return self.yield_frame_from_rtsp()

    def yield_frame_from_image(self):
        yield cv2.imread(self.path)

    def yield_frame_from_video(self):
        with video_capture(self.path) as cap:
            while True:
                ret, img = cap.read()
                if ret:
                    yield img
                else:
                    # raise RuntimeError("Failed to capture image")
                    break

    def yield_frame_from_rtsp(self):
        with video_capture(self.path) as cap:
            while True:
                ret, img = cap.read()
                if ret:
                    yield img
                else:
                    # raise RuntimeError("Failed to capture image")
                    break

    def yield_frame_from_image_dir(self):
        for file in os.listdir(self.path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = self.path + file if self.path.endswith('/') else self.path + '/' + file
                yield cv2.imread(file_path)




