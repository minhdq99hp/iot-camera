from frame_generator import *
import cv2
import sys
import csv

import minh_custom_keras_yolo3.yolo as y
import tensorflow as tf
from datetime import timedelta

yolo = y.YOLO()
graph = tf.get_default_graph()

from datetime import datetime

videopath = sys.argv[1]

filename = os.path.basename(videopath)

filebasename, file_extension = os.path.splitext(filename)

datetime_object = datetime.strptime(filebasename[7:], '%d%m%Y%H%M%S')

datetime_str = datetime_object.strftime('%d/%m/%Y %H:%M:%S')

print(f'datetime: {datetime_str}')

my_frame_number = 3600

vid = cv2.VideoCapture(videopath)

width = 640
height = width // 16 * 9

i = 0

csv_file = open('output.csv', mode='w')
fieldnames = ['datetime', 'count']
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

fps = vid.get(cv2.CAP_PROP_FPS)
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

print(fps)
print(total_frames)

while vid.isOpened():
    ret, frame = vid.read()
    i += 1
    print(i)
    if ret:
        if i % fps == 0:
            # frame = cv2.resize(frame, (width, height))
            detected, detection_info = yolo.detect_person_cv2(frame)

            datetime_str = datetime_object.strftime('%d/%m/%Y %H:%M:%S')
            datetime_object += timedelta(seconds=1)

            # cv2.imshow("Show", detected)

            csv_writer.writerow([datetime_str, str(detection_info['count_boxes'])])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

csv_file.close()
vid.release()
cv2.destroyAllWindows()
