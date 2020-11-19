# coding=utf-8

# Plot without display
# must put before using any display backend


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
from tensorflow.python.saved_model import tag_constants
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 0))  # second
RES_URL = os.getenv('RES_URL', "http://127.0.0.1:8080/video_feed")
ORION_TASK_ID = os.getenv('ORION_TASK_ID', "obj_0001")
OBJECT_OUTPUT_PORT = os.getenv('OBJECT_OUTPUT_PORT', "8080")

batchsize = 1


inputFrame = None
wait_time = 10

outputFrame = None
lock = threading.Lock()
loginfo = "Video Stream is Running..."




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
            #send_result.delay(loginfo)
            time.sleep(wait_time)
            pass


def run_detection(img,model):
    global outputFrame, lock, loginfo
    input_size=416
    img = tf.constant(img)
    pred_bbox = model(img)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

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


def object_detection(model):
    url = parse.urlparse(RES_URL)
    h = http.client.HTTPConnection(url.netloc)
    h.request('GET',url.path)
    res = h.getresponse()

    global inputFrame, outputFrame, lock, loginfo

    image_lists = []
    print("Begin to get video frames...")

    while True:
        try:
            image_lists.append(inputFrame)
            if len(image_lists) == batchsize:
                # results = yolo.inference(image_lists)
                image = preprocess(image_lists[0])
                results = run_detection(image,model)
                if not results:
                    results = "No detection"
                # print("results: ", results[0][0][:5])
                
                #draw box
                print('draw_box')
                image, results = draw_bbox(image_lists[0], results)
                cv2.imwrite('demo.jpg',image)
                # image = image_lists
                # print(results)
                flag, encodedImage = cv2.imencode(".jpg", image)
                
                my_stringIObytes = BytesIO(encodedImage)
                my_stringIObytes.seek(0)
                

                with lock:
                    outputFrame = my_stringIObytes.read()

                r = ORION_TASK_ID + ' @@ ' + str(results)
                print(r)
                #send_result.delay(r)
                time.sleep(FRAME_INTERVAL)
                image_lists = []

        except Exception as e:
            loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
            print(loginfo)
            loginfo = ORION_TASK_ID + ' @@ ' + loginfo
            #send_result.delay(loginfo)
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
    print("loading weights and engine file...")
    saved_model_loaded = tf.saved_model.load('model/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    print("Weights and engine file loaded.")
    t1 = threading.Thread(target=get_frame)
    t1.daemon = True
    t1.start()

    # wait 3 seconds to get the frames ready
    time.sleep(3)

    # start a thread that will perform motion detection
    t = threading.Thread(target=object_detection,args = (infer,))
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', threaded=True, debug=True, port="8090", use_reloader=False)
