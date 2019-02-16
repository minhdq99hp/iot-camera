import argparse
import cv2
import os
from pathlib import Path
from keras.utils.data_utils import get_file
import numpy as np
import dlib
from age_gender_estimation.wide_resnet import WideResNet
from PIL import Image
import minh_custom_keras_yolo3.yolo as y
from frame_generator import FrameGenerator

# Disable warning: "...TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Age and Gender Estimation Model
pretrained_age_gender_model = "https://github.com/yu4u/age_gender_estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
img_size = 64
margin = 0.4
ag_model = None
face_detector = None

# YOLOv3 Model
yolo = y.YOLO()


def get_args():
    parser = argparse.ArgumentParser(description="This script demo the IotCamera project of Minh DS. ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--weight_file", type=str, default=None,
    #                     help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--age_gender", type=int, default=1, help="estimating age & gender")
    parser.add_argument("--person", type=int, default=1, help="detecting person")
    parser.add_argument("--motion", type=int, default=1, help="detecting motion")

    # parser.add_argument("--image_dir", type=str, default=None,
    #                     help="target image directory; if set, images in image_dir are used instead of webcam")

    parser.add_argument("--input", type=str, default=None, help="input image or video directory, "
                                                                "if not given, the script will run with webcam instead")
    parser.add_argument("--output", type=str, default=None, help="output image or video directory, "
                                                                 "in order to save the file")
    args = parser.parse_args()
    return args


def draw_label_age_gender(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def draw_label_age_gender_faces(img, detected, age_and_gender):

    for i, d in enumerate(detected):
        label = "{}, {}".format(age_and_gender[i][0], "M" if age_and_gender[i][1] == 1 else "F")
        draw_label_age_gender(img, (d.left(), d.top()), label)


def draw_bounding_box_faces(img, detected):
    img_h, img_w, _ = np.shape(img)

    if len(detected) > 0:
        for i, d in enumerate(detected):
            # draw bounding box
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def proceed_age_gender_estimation(img):
    global ag_model, face_detector, img_size, margin

    # resize img
    h, w, _ = img.shape
    r = 640 / max(w, h)
    input_img = cv2.resize(img, (int(w * r), int(h * r)))

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = face_detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    age_and_gender = []

    if len(detected) > 0:
        for i, d in enumerate(detected):

            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            # draw bounding box
            cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)

            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = ag_model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()


        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "M" if predicted_genders[i][0] < 0.5 else "F")
            draw_label_age_gender(input_img, (d.left(), d.top()), label)

            age_and_gender.append((int(predicted_ages[i]), 1 if predicted_genders[i][0] < 0.5 else 0))

    return input_img, detected, age_and_gender


def proceed_person_detection(img):
    # convert OpenCV Image to PIL Image
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    detected = yolo.detect_image(pil_im)

    # convert PIL Image to OpenCV Image
    detected = np.array(detected)
    # Convert RGB to BGR
    detected = detected[:, :, ::-1].copy()

    return detected


def main():
    print("NO")
    # global face_detector, ag_model, img_size
    #
    # global yolo
    #
    # # get and check arguments
    # args = get_args()
    # mode = -1
    # # 0: image
    # # 1: video
    # # 2: image_dir
    # # 3: rtsp
    #
    # # Check input
    # if args.input is not None:
    #     if os.path.exists(args.input):
    #         if args.input.lower().endswith((".jpg", ".jpeg", ".png")):
    #             mode = 0
    #         elif args.input.lower().endswith((".mp4", ".avi")):
    #             mode = 1
    #         elif args.input.lower().startswith("rtsp://"):
    #             mode = 3
    #         elif os.path.isdir(args.input):
    #             mode = 2
    #         else:
    #             print("Input is not supported !");
    #             return
    #     else:
    #         print("Input not found !")
    #         return
    # else:
    #     print("No Input !")
    #     return
    #
    # # Check output
    # if args.output is not None:
    #     if mode == 0:
    #         if not args.output.lower().endswith((".jpg", ".jpeg", ".png")):
    #             args.output += ".jpg"
    #     elif mode == 1:
    #         if not args.output.lower().endswith((".mp4", ".avi")):
    #             args.output += ".mp4"
    #     elif mode == 2:
    #         if not os.path.isdir(args.output):
    #             print("Output must be a directory for mode 2!")
    #             return
    #     # elif mode == 3:
    # else:
    #     print("Notice: No Output")
    #
    # frameGenerator = FrameGenerator(mode, args.input)
    #
    # # Load models
    # if args.age_gender:
    #     # for face detection
    #     face_detector = dlib.get_frontal_face_detector()
    #
    #     # load age_gender_model and weights
    #     ag_model = WideResNet(img_size, depth=16, k=8)()
    #
    #     weight_file = get_file("weights.28-3.73.hdf5", pretrained_age_gender_model,
    #                            cache_subdir="age_gender_estimation/pretrained_models",
    #                            file_hash=modhash,
    #                            cache_dir=str(Path(__file__).resolve().parent))
    #     ag_model.load_weights(weight_file)
    #
    # if args.person:
    #     # load model and weights
    #     yolo = y.YOLO()
    #     pass
    #
    # for frame in frameGenerator.yield_frame():
    #     proceed_frame(frame, mode, args)

    # END HERE
    #
    # if input_path is None:
    #     # using camera
    #     cap = cv2.VideoCapture(0)
    #
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #
    #     # video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    #     video_fps = cap.get(cv2.CAP_PROP_FPS)
    #     video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #
    #     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), video_fps, video_size) \
    #         if output_path else None
    #
    #     while cap.isOpened():
    #         ret, origin_frame = cap.read()
    #
    #         frame = origin_frame.copy()
    #         if ret:
    #             output_person_img = None
    #             if args.age_gender:
    #                 output_ag_img, detected, age_and_gender = proceed_age_gender_estimation(frame)
    #
    #                 draw_bounding_box_faces(origin_frame, detected)
    #                 draw_label_age_gender_faces(origin_frame, detected, age_and_gender)
    #             if args.person:
    #                 # convert OpenCV Image to PIL Image
    #                 cv2_im = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    #                 pil_im = Image.fromarray(cv2_im)
    #                 output_person_img = yolo.detect_image(pil_im)
    #             if args.motion:
    #                 pass
    #
    #             if output_path:
    #                 out.write(origin_frame)
    #
    #             if args.person:
    #                 # convert PIL Image to OpenCV Image
    #                 output_person_img = np.array(output_person_img)
    #                 # Convert RGB to BGR
    #                 output_person_img = output_person_img[:, :, ::-1].copy()
    #                 cv2.imshow("person", output_person_img)
    #
    #             cv2.imshow("result", origin_frame)
    #
    #             # cv2.imshow("age and gender", output_ag_img)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #
    #     cap.release()
    #     if output_path:
    #         out.release()
    #     cv2.destroyAllWindows()
    #
    # else:
    #     if os.path.isfile(input_path):
    #         if input_path.lower().endswith((".jpg", ".png", ".jpeg")):
    #             # work with 1 image
    #             return
    #         elif input_path.lower().endswith((".mp4", ".avi")):
    #             # work with 1 video
    #             # using camera
    #             cap = cv2.VideoCapture(input_path)
    #
    #             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #
    #             # video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    #             video_fps = cap.get(cv2.CAP_PROP_FPS)
    #             video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #
    #             out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), video_fps, video_size) \
    #                 if output_path else None
    #
    #             while cap.isOpened():
    #                 ret, origin_frame = cap.read()
    #
    #                 frame = origin_frame.copy()
    #                 if ret:
    #                     output_person_img = None
    #                     if args.age_gender:
    #                         output_ag_img, detected, age_and_gender = proceed_age_gender_estimation(frame)
    #                         draw_bounding_box_faces(origin_frame, detected)
    #                         draw_label_age_gender_faces(origin_frame, detected, age_and_gender)
    #                     if args.person:
    #                         # convert OpenCV Image to PIL Image
    #                         cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         pil_im = Image.fromarray(cv2_im)
    #                         output_person_img = yolo.detect_image(pil_im)
    #                     if args.motion:
    #                         pass
    #
    #                     if args.person:
    #                         # convert PIL Image to OpenCV Image
    #                         output_person_img = np.array(output_person_img)
    #                         # Convert RGB to BGR
    #                         output_person_img = output_person_img[:, :, ::-1].copy()
    #
    #                         cv2.imshow("person", output_person_img)
    #
    #                     cv2.imshow("result", origin_frame)
    #
    #                     if output_path:
    #                         out.write(origin_frame)
    #
    #                     # cv2.imshow("age and gender", output_ag_img)
    #                     if cv2.waitKey(1) & 0xFF == ord('q'):
    #                         break
    #
    #
    #
    #             cap.release()
    #             if output_path:
    #                 out.release()
    #             cv2.destroyAllWindows()
    #
    #         else:
    #             print("Unsupported Video file !")
    #     elif os.path.isdir(input_path):
    #     # work with images directory
    #         return
    #     else:
    #         raise FileNotFoundError


if __name__ == '__main__':
    main()
