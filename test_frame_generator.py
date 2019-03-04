from frame_generator import *
import cv2
import sys

videopath = sys.argv[1]
outputpath = sys.argv[2]

frameGenerator = FrameGenerator(StreamMode.VIDEO, videopath)
fps = int(frameGenerator.vid_fps)
size = frameGenerator.vid_size
cc = int(frameGenerator.vid_cc)

print(size)

width = 640
height = width // 16 * 9

print(cc)

outputVid = cv2.VideoWriter(outputpath, cc, fps, (width, height))
i = 0
for frame in frameGenerator.yield_frame():
    i += 1

    frame = cv2.resize(frame, (width, height))

    # matrix = cv2.getRotationMatrix2D((width//2, height//2), -10, 1)
    #
    # rotatedFrame = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)

    if i % 3 == 0:
        outputVid.write(frame)
        cv2.imshow("Show", frame)
    # cv2.imshow("Rotated", rotatedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('DONE !')

outputVid.release()

cv2.destroyAllWindows()
