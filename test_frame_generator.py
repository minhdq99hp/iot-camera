from frame_generator import *
import cv2

path_2 = '/home/minhdq99hp/Desktop/1.mp4'
path_1 = '/home/minhdq99hp/Desktop/pic/'
path_0 = '/home/minhdq99hp/Desktop/pic/1.JPG'


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


frameGenerator = FrameGenerator(2, path_2)

# im = cv2.imread(path)
# cv2.imshow("Show", im)
# cv2.waitKey(0)

for frame in frameGenerator.yield_frame():
    cv2.imshow("Show", frame)
    key = cv2.waitKey(30) # 30 FPS
    if key == 27:
        break


cv2.destroyAllWindows()