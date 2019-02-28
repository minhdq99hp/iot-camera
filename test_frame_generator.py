from frame_generator import *
import cv2
import sys

videopath = sys.argv[1]
outputpath = '~/Desktop/output.avi'

frameGenerator = FrameGenerator(StreamMode.VIDEO, videopath)
fps = int(frameGenerator.vid_fps)
size = frameGenerator.vid_size
cc = int(frameGenerator.vid_cc)

print(size)

width = 640
height = width // 16 * 9

print(cc)

outputVid = cv2.VideoWriter(outputpath, cc, fps, (width, height))

for frame in frameGenerator.yield_frame():
    frame = cv2.resize(frame, (width, height))

    matrix = cv2.getRotationMatrix2D((width//2, height//2), -10, 1)

    rotatedFrame = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)

    outputVid.write(rotatedFrame)

    cv2.imshow("Show", frame)
    cv2.imshow("Rotated", rotatedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('DONE !')

outputVid.release()

cv2.destroyAllWindows()
