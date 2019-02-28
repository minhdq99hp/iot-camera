import sys
sys.path.append('/home/minhdq99hp/iot-camera')
sys.path.append('/home/minhdq99hp/iot-camera/minh_custom_keras_yolo3')

from frame_generator import FrameGenerator, StreamMode
from flask import request, Response
import flask
import os
import cv2

# CONSTANT
proceeded_data_path = "static/proceeded_data"
uploaded_data_path = "static/uploaded_data"
show_frame = True

rtsp_url = 'http://83.211.71.120:8084/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER'

app = flask.Flask(__name__)

fg = FrameGenerator(StreamMode.RTSP, rtsp_url)




def proceed_video_streaming(frame_generator, streaming_id):
    filename = f'{streaming_id}.jpg'
    filepath = os.path.join(proceeded_data_path, filename)


    for frame in frame_generator.yield_frame():
        # detected, frame_info = detect_person(frame)

        cv2.imwrite(filepath, frame)
        if show_frame:
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        binary_file = open(filepath, 'rb').read()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + binary_file + b'\r\n')

    cv2.destroyAllWindows()


@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return Response(proceed_video_streaming(fg, 'rtsp_streaming_id'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
