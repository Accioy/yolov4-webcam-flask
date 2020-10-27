# coding=utf-8

# Plot without display
# must put before using any display backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cv2

from io import BytesIO
import math
import os
import time

from cam_test import MJPEGClient
from result_publisher.result_publisher import send_result
import http.client
from urllib import parse
import numpy as np

import threading
from flask import Flask, render_template, Response
from utils import *
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 1))  # second
RES_URL = os.getenv('RES_URL', "http://192.168.12.150:8090/stream.mjpg")
ORION_TASK_ID = os.getenv('ORION_TASK_ID', "obj_0001")
OBJECT_OUTPUT_PORT = os.getenv('OBJECT_OUTPUT_PORT', "8080")

batchsize = 1 # Can only be 1


inputFrame = None
wait_time = 10

outputFrame = None
lock = threading.Lock()
loginfo = "Video Stream is Running..."

print("loading weights and engine file...")


def load_model_lite(path):
    #input_size=416
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details[0]['index'], output_details



interpreter,inputs,output_details = load_model_lite('model/yolov4-416-fp16.tflite')

print("Weights and engine file loaded.")

app = Flask(__name__)

@app.route('/')
def index():
    global loginfo
    """Video streaming home page."""
    return render_template('index.html', loginfo = loginfo)


def get_frame():
    global inputFrame, loginfo
    while True:
        try:
            for jpegdata in MJPEGClient(RES_URL):
                response = BytesIO(jpegdata)
                img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
                inputFrame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        except Exception as e:
            loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
            print(loginfo)
            loginfo = ORION_TASK_ID + ' @@ ' + loginfo
            send_result.delay(loginfo)
            time.sleep(wait_time)
            pass


def run_detection(img):
    global interpreter,inputs,output_details, outputFrame, lock, loginfo
    input_size=416
    interpreter.set_tensor(inputs, img)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))


    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox


def object_detection():
    url = parse.urlparse(RES_URL)
    h = http.client.HTTPConnection(url.netloc)
    h.request('GET',url.path)
    res = h.getresponse()

    global inputFrame, outputFrame, lock, loginfo, yolo

    image_lists = []
    print("Begin to get video frames...")

    while True:
        try:
            image_lists.append(inputFrame)
            if len(image_lists) == batchsize:
                # results = yolo.inference(image_lists)
                image = preprocess(image_lists[0])
                results = run_detection(image)

                #draw box
                print('draw_box')
                image, results = draw_bbox(image_lists[0], results)
                if len(results)==0:
                        results='No detection'
                #print(results)
                cv2.imwrite('demo.jpg',image)
                # image = image_lists
                fig = plt.figure()
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                my_stringIObytes = BytesIO()

                # save fig without white margin
                plt.axis('off')

                height, width, channels = image.shape
                fig.set_size_inches(width / fig.dpi, height / fig.dpi)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(my_stringIObytes, dpi=fig.dpi)

                my_stringIObytes.seek(0)
                plt.close(fig)

                with lock:
                    outputFrame = my_stringIObytes.read()

                r = ORION_TASK_ID + ' @@ ' + str(results)
                print('sending results...')
                print(r)
                send_result.delay(r)
                print('results send done')
                time.sleep(FRAME_INTERVAL)
                image_lists = []

        except Exception as e:
            loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
            print(loginfo)
            loginfo = ORION_TASK_ID + ' @@ ' + loginfo
            send_result.delay(loginfo)
            time.sleep(wait_time)
            pass


def generate():
    """Video streaming generator function."""
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # # encode the frame in JPEG format
            # (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            #
            # ensure the frame was successfully encoded
            # if not flag:
            #     continue
            encodedImage = outputFrame
        # yield the output frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # start a thread that will get frames
    t1 = threading.Thread(target=get_frame)
    t1.daemon = True
    t1.start()

    # wait 3 seconds to get the frames ready
    time.sleep(3)

    # start a thread that will perform motion detection
    t = threading.Thread(target=object_detection)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', threaded=True, debug=True, port=OBJECT_OUTPUT_PORT, use_reloader=False)
