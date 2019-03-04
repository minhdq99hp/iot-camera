from frame_generator import *
import cv2
import sys
from datetime import datetime

videopath = sys.argv[1]

filename = os.path.basename(videopath)

filebasename, file_extension = os.path.splitext(filename)

datetime_object = datetime.strptime(filebasename[7:], '%m%Y%H%M%S')

datetime_str = datetime_object.strftime('??/%m/%Y %H:%M:%S')

print(f'datetime: {datetime_str}')

my_frame_number = 3600


outputpath = ''
has_output = False
if len(sys.argv) == 3:
    outputpath = sys.argv[2]
    has_output = True

frameGenerator = FrameGenerator(StreamMode.VIDEO, videopath)

total_frames = frameGenerator.total_frames
fps = int(frameGenerator.vid_fps)
size = frameGenerator.vid_size
cc = int(frameGenerator.vid_cc)
width = 640
height = width // 16 * 9

print(f'Total Frames: {total_frames}')
print(f'FPS: {fps}')

print(size)
print(width, height)



if has_output:
    outputVid = cv2.VideoWriter(outputpath, cc, fps, (width, height))
i = 0
for frame in frameGenerator.yield_frame():
    i += 1

    frame = cv2.resize(frame, (width, height))

    # matrix = cv2.getRotationMatrix2D((width//2, height//2), -10, 1)
    #
    # rotatedFrame = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)

    if i % 3 == 0:
        if has_output:
            outputVid.write(frame)
        cv2.imshow("Show", frame)

    if cv2.waitKey(1000//6) & 0xFF == ord('q'):
        break

print('DONE !')

outputVid.release()

cv2.destroyAllWindows()
