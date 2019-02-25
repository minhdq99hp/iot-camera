import sys
sys.path.append('/home/minhdq99hp/iot-camera')
sys.path.append('/home/minhdq99hp/iot-camera/minh_custom_keras_yolo3')

import os
import cv2
import numpy as np
from PIL import Image
import minh_custom_keras_yolo3.yolo as y
import tensorflow as tf
from frame_generator import FrameGenerator, StreamMode
from urllib.request import unquote
import time

import uuid
import flask
from flask import Response, request, url_for, jsonify, render_template, send_from_directory, redirect, json
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

# CONSTANT
proceeded_data_path = "static/proceeded_data"
uploaded_data_path = "static/uploaded_data"
show_frame = True

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


def proceed(filename, output_type):
    # INPUT_PATH
    input_path = os.path.join(uploaded_data_path, filename)

    # OUTPUT_PATH
    output_filename = f'{str(uuid.uuid4())}_{filename}'
    output_path = os.path.join(proceeded_data_path, output_filename)

    # OUTPUT_INFO
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

        detection_info['frames'].append(frame_info)
        detection_info['count_frames'] += 1
        detection_info['time_interval'] += frame_info['time_interval']

    # PROCEED VIDEO
    elif filename.lower().endswith(('.mp4', '.avi')):
        # USING FRAME GENERATOR
        frame_generator = FrameGenerator(StreamMode.VIDEO, input_path)

        output_file = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                      frame_generator.vid_fps, frame_generator.vid_size)

        for frame in frame_generator.yield_frame():
            detected, frame_info = detect_person(frame)

            if show_frame:
                cv2.imshow('detected', detected)

            output_file.write(detected)

            detection_info['frames'].append(frame_info)
            detection_info['time_interval'] += frame_info['time_interval']
            detection_info['count_frames'] += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        output_file.release()
        cv2.destroyAllWindows()

        # USING OFFICIAL SOLUTION
        # vid = cv2.VideoCapture(input_path)
        #
        # fps = vid.get(cv2.CAP_PROP_FPS)
        # size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #
        # output_file = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        #
        # while vid.isOpened():
        #     ret, frame = vid.read()
        #     if ret:
        #         detected, frame_info = detect_person(frame)
        #
        #         if show_frame:
        #             cv2.imshow('detected', detected)
        #
        #         output_file.write(detected)
        #
        #         detection_info['frames'].append(frame_info)
        #         detection_info['time_interval'] += frame_info['time_interval']
        #         detection_info['count_frames'] += 1
        #
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     else:
        #         break
        #
        # output_file.release()
        # vid.release()
        # cv2.destroyAllWindows()

        # END PROCESS



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
    # cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img)

    with graph.as_default():
        detected, detection_info = yolo.detect_person(pil_im)

    # convert PIL Image to OpenCV Image
    detected = np.asarray(detected)
    # Convert RGB to BGR
    # detected = detected[:, :, ::-1].copy()

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
        print_header('A NEW REQUEST COMING !')
        # GETTING DATA
        print_header('GETTING DATA')
        f = request.files['file_input']
        filename = secure_filename(f.filename)
        output_type = request.form.get("output_type")

        # SAVE FILE TO LOCAL
        print_header('SAVE FILE TO LOCAL')
        filepath = os.path.join(uploaded_data_path, filename)
        f.save(filepath)

        # PROCEED THE FILE
        print_header('PROCEED FILE')
        result = proceed(filename, output_type)

        output_file_path = os.path.join(proceeded_data_path, result['output_filename'])

        if output_type == 'output_file':  # RETURN PROCEEDED FILE
            try:
                print_header('RETURN PROCEEDED FILE')
                # return send_file(output_file_path, attachment_filename=result["output_filename"])
                return send_from_directory(proceeded_data_path, result['output_filename'])
            except Exception as e:
                print(e)
        elif output_type == 'output_json':  # RETURN JSON FILE
            return jsonify(result)

        else:  # RETURN HTML PAGE
            print_header('OUTPUT_HTML')
            if output_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # return render_template('index.html',
                #                        result_file=Markup(f'<img class="img-fuild" '
                #                                           f'style="max-width:100%; height:auto;" '
                #                                           f'src="{output_file_path}" alt="Result">'))
                return redirect(output_file_path)
            elif output_file_path.lower().endswith(('.mp4', '.avi')):
                # return render_template('index.html',
                #                        result_file=Markup(f'<video style="max-width:100%; height:auto;" controls>'
                #                                           f'<source src="{output_file_path}" type="video/mp4">'
                #                                           f'Sorry, your browser doesn\'t support embedded videos.'
                #                                           f'</video>'))
                return redirect(output_file_path)


def proceed_webcam_streaming(frame_generator, streaming_id):

    for frame in frame_generator.yield_frame():

        filename = f'{streaming_id}.jpg'
        filepath = os.path.join(proceeded_data_path, filename)

        detected, frame_info = detect_person(frame)

        cv2.imwrite(filepath, detected)

        binary_file = open(filepath, 'rb').read()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + binary_file + b'\r\n')


def proceed_video_streaming(frame_generator, streaming_id):
    filename = f'{streaming_id}.jpg'
    filepath = os.path.join(proceeded_data_path, filename)


    for frame in frame_generator.yield_frame():
        detected, frame_info = detect_person(frame)

        cv2.imwrite(filepath, detected)
        if show_frame:
            cv2.imshow("Frame", detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        binary_file = open(filepath, 'rb').read()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + binary_file + b'\r\n')

    cv2.destroyAllWindows()


@app.route('/streaming', methods=['GET', 'POST'])
def streaming():
    if request.method == 'GET':
        # TEST RTSP. DELETE THIS PART WHEN DONE.

        frame_generator = FrameGenerator(StreamMode.RTSP, 'http://127.0.0.1:8554')
        streaming_id = '7ddc8ae0-f147-40da-81fd-28a72f79e6b3'
        return Response(proceed_webcam_streaming(frame_generator, streaming_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        # END PART

        data = request.get_json()
        print(data)

        if data is None:
            streaming_id = uuid.uuid4()
            # TEST ON WEBCAM
            print_header('START STREAMING WEBCAM')
            frame_generator = FrameGenerator(StreamMode.WEBCAM)
        elif 'file_path' in data:
            streaming_id = data['streaming_id']
            # START STREAMING VIDEO
            print_header('START STREAMING VIDEO')

            # GETTING UPLOADED FILE
            filepath = ""
            for p in os.listdir(uploaded_data_path):
                if p.startswith(streaming_id):
                    filepath = os.path.join(uploaded_data_path, p)
                    break
            if filepath == "":
                raise Exception('Can\'t find the streaming file !')

            frame_generator = FrameGenerator(StreamMode.VIDEO, filepath)

        if 'rtsp_url' in request.args:
            streaming_id = request.args['streaming_id']
            # START STREAMING RTSP
            print_header('START STREAMING RTSP')
            rtsp_url = data['rtsp_url']

            frame_generator = FrameGenerator(StreamMode.RTSP, rtsp_url)


        return Response(proceed_webcam_streaming(frame_generator, streaming_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif request.method == 'POST':
        result = {}

        print_header('A NEW REQUEST COMING !')
        # GENERATING A NEW STREAMING ID
        streaming_id = uuid.uuid4()
        result['streaming_id'] = streaming_id

        if 'file_input' in request.files:
            # GETTING DATA
            print_header('GETTING DATA')
            f = request.files['file_input']
            filename = secure_filename(f.filename)

            # SAVE FILE TO LOCAL. Filename = streaming_id + extension
            print_header('SAVE FILE TO LOCAL')
            filepath = os.path.join(uploaded_data_path, f'{streaming_id}{os.path.splitext(filename)[1]}')
            f.save(filepath)

            result['file_path'] = filepath

        elif 'rtsp_url' in request.form:
            result['rtsp_url'] = request.form['rtsp_url']


        return jsonify(result)


if __name__ == "__main__":
    print_header('LOADING YOLO')
    load_models()

    print_header('CLEAN STATIC FOLDER')
    clean_static_folder()

    app.run(debug=False, threaded=True)
