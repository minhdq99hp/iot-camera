from PIL import Image
import numpy as np
import cv2


def print_header(s):
    s = s.upper().strip()

    len_header = 30

    if len(s) >= len_header:
        print(s)
    else:
        start_pos = len_header // 2 - len(s) // 2
        print(f'\n+{"-" * start_pos}{s}{"-" * (len_header-start_pos-len(s))}+\n')


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2_img)


def pil_to_cv2(pil_img):
    return np.asarray(pil_img)


def has_video_extension(file_path):
    return file_path.lower().endswith(('.mp4', '.avi', '.h264'))

def has_image_extension(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))
