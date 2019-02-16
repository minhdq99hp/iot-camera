import sys
sys.path.append('/home/minhdq99hp/iot-camera')
sys.path.append('/home/minhdq99hp/iot-camera/minh_custom_keras_yolo3')

import os
import cv2
import numpy as np
from PIL import Image
import minh_custom_keras_yolo3.yolo as y
import tensorflow as tf
from frame_generator import FrameGenerator

import uuid
import flask
from flask import request, jsonify, render_template, send_file, make_response, Markup, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

# CONSTANT
proceeded_data_path = "static/proceeded_data"
uploaded_data_path = "static/uploaded_data"
show_frame = False

yolo = None
graph = None



def print_header(s):
    s = s.upper().strip()

    len_header = 30

    if len(s) >= len_header:
        print(s)
    else:
        start_pos = len_header // 2 - len(s) // 2
        print(f'\n+{"-" * start_pos}{s}{"-" * (len_header-start_pos-len(s))}+\n')



def proceed(filename, output_json, output_file):
    input_path = os.path.join(uploaded_data_path, filename)

    output_filename = f'{str(uuid.uuid4())}_{filename}'

    output_path = os.path.join(proceeded_data_path, output_filename)


    detection_info = {'frames': [],
                      'time_interval': 0,
                      'count_frames': 0,
                      'output_path': output_path,
                      'output_filename': output_filename}

    # PROCEED IMAGE
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_img = cv2.imread(input_path)

        detected, frame_info = detect_person(input_img)

        cv2.imwrite(os.path.join(proceeded_data_path, output_filename), detected)

    # PROCEED VIDEO
    elif filename.lower().endswith(('.mp4', '.avi')):
        vid = cv2.VideoCapture(input_path)

        fps = vid.get(cv2.CAP_PROP_FPS)
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        output_file = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

        # USING FRAME GENERATOR
        # frameGenerator = FrameGenerator(2, input_path)
        # for frame in frameGenerator.yield_frame():
        #     detected, detection_info = detect_person(frame)
        #
        #     output_file.write(detected)

        # USING OFFICIAL SOLUTION
        while vid.isOpened():
            ret, frame = vid.read()
            frame = frame.rotate(-15)
            if ret:
                detected, frame_info = detect_person(frame)

                if show_frame:
                    cv2.imshow('detected', detected)

                output_file.write(detected)

                detection_info['frames'].append(frame_info)
                detection_info['time_interval'] += frame_info['time_interval']
                detection_info['count_frames'] += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        output_file.release()
        vid.release()
        cv2.destroyAllWindows()


    detection_info["output_filename"] = output_filename

    return detection_info

# # detection_info (of a single frame):
# time_interval
# count_boxes
# output_filename
# boxes
#   label
#   box
#   score

# detection_info (of a video):
# time_interval
# count_frames
# frames


def detect_person(img):
    # convert OpenCV Image to PIL Image
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    with graph.as_default():
        detected, detection_info = yolo.detect_person(pil_im)

    # convert PIL Image to OpenCV Image
    detected = np.array(detected)
    # Convert RGB to BGR
    detected = detected[:, :, ::-1].copy()

    return detected, detection_info


def load_models():
    global yolo
    global graph
    yolo = y.YOLO()
    graph = tf.get_default_graph()



def clean_static_folder():
    # VERY DANGEROUS
    if len(os.listdir(proceeded_data_path)) > 0:
        os.system(f'rm {os.path.join(proceeded_data_path, "*")}')

    if len(os.listdir(uploaded_data_path)) > 0:
        os.system(f'rm {os.path.join(uploaded_data_path, "*")}')


print_header('LOADING FLASK APP')
app = flask.Flask(__name__)
Bootstrap(app)


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        # GETTING DATA
        print_header('GETTING DATA')
        f = request.files['fileInput']
        filename = secure_filename(f.filename)
        output_json = request.form.get("output_json")
        output_file = request.form.get("output_file")
        
        
        # SAVE FILE TO LOCAL
        print_header('SAVE FILE TO LOCAL')
        filepath = os.path.join(uploaded_data_path, filename)
        f.save(filepath)

        # PROCEED THE FILE
        print_header('PROCEED FILE')
        result = proceed(filename, output_json, output_file)


        output_file = True

        if output_file:  # RETURN PROCEEDED FILE
            try:
                return send_file(os.path.join(proceeded_data_path, result["output_filename"]),
                                 attachment_filename=result["output_filename"])
            except Exception as e:
                print(e)
        elif output_json:  # RETURN JSON FILE
            try:
                del result["output_file"]

            except KeyError:
                print("Key 'output_file' not found !")

            return jsonify(result)
        else:  # RETURN HTML PAGE
            return "nothing"


# @app.route('/proceeded_data/<string:filename>', methods=['GET'])
# def proceeded_data(filename):
#     file_path = os.path.join(proceeded_data_path, filename)
#
#     if os.path.exists(file_path):
#         return Markup('<img style="width=100%;" src="' + file_path + '"/>')
#
#     return "NOT YET !"


if __name__ == "__main__":
    print_header('LOADING YOLO')
    load_models()

    print_header('CLEAN STATIC FOLDER')
    clean_static_folder()

    app.run(debug=False, threaded=True)