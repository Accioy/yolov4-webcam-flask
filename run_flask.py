# coding=utf-8

# Plot without display
# must put before using any display backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# import tensorflow as tf
import cv2

from io import BytesIO
import math
import os
import time

# from cam_test import MJPEGClient
from local_cam_test import local_cam
from result_publisher.result_publisher import send_result
import http.client
from urllib import parse
import numpy as np

import threading
from flask import Flask, render_template, Response
# from utils import preprocess,draw_bbox,filter_boxes
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 1))  # second
# RES_URL = os.getenv('RES_URL', "http://192.168.12.150:8090/stream.mjpg")
ORION_TASK_ID = os.getenv('ORION_TASK_ID', "IMA_0001")
OBJECT_OUTPUT_PORT = os.getenv('OBJECT_OUTPUT_PORT', "8080")

batchsize = 1
# config_file = "config/yolov3.txt"

inputFrame = None
wait_time = 3

outputFrame = None
lock = threading.Lock()
loginfo = "Video Stream is Running..."

# print("loading weights and engine file...")
# yolo = Yolov3(batchsize, config_file)
# print("Weights and engine file loaded.")

def load_model_lite(path):
    #input_size=416
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details[0]['index'], output_details



# interpreter,inputs,output_details = load_model_lite('model/yolov4-416-fp32.tflite')
interpreter,inputs,output_details = None,None,None

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
			# for jpegdata in MJPEGClient(RES_URL):
			# 	response = BytesIO(jpegdata)
			# 	img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
			# 	inputFrame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
			# 	inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2RGB)
			# 	inputFrame = cv2.resize(original_image, (input_size, input_size))
			# 	inputFrame = inputFrame/255.

			for frame in local_cam():
				inputFrame = frame
		except Exception as e:
			loginfo = 'error while get_frame, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
			print(loginfo)
			loginfo = ORION_TASK_ID + ' @@ ' + loginfo
			send_result.delay(loginfo)
			time.sleep(wait_time)
			pass






def run_infer():
	global interpreter,inputs,output_details, inputFrame, outputFrame, lock, loginfo

	image_lists = []
	print("Begin to get video frames...")
	while True:
		try:
			if inputFrame is not None:
				image_lists.append(inputFrame)
			if len(image_lists) == batchsize: #1
				# image = preprocess(inputFrame)
				# pre_bbox = run_detection(image)
				# image = draw_bbox(inputFrame, pred_bbox)
				image = image_lists[0].copy()
				flag, encodedImage = cv2.imencode(".jpg", image)
				encodedImage = np.array(encodedImage).tobytes()
				with lock:
					outputFrame = encodedImage

				results = ORION_TASK_ID
				send_result.delay(results)
				time.sleep(FRAME_INTERVAL)
				image_lists = []
		except Exception as e:
			raise e
			loginfo = 'error while run_infer, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
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
			   b'Content-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n')


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
	t = threading.Thread(target=run_infer)
	t.daemon = True
	t.start()

	app.run(host='127.0.0.1', threaded=True, debug=True, port=5000)
