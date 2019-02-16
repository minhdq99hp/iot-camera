import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


img_path = 'test.jpg'

image = Image.open(img_path)

r_image = YOLO().detect_image(image)
r_image.show()
r_image.save("predict.jpg", "JPEG")
