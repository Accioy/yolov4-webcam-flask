import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import cv2
from utils import preprocess,draw_bbox,filter_boxes
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession



def load_model_lite(path):
	#input_size=416
	interpreter = tf.lite.Interpreter(model_path=path)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	return interpreter, input_details[0]['index'], output_details



interpreter,inputs,output_details = load_model_lite('model/yolov4-416-chg0.tflite')
print('input_details:',inputs)

print('output_details:',output_details)
print(len(output_details))



def run_detection(img):
	global interpreter,inputs,output_details
	input_size=416
	interpreter.set_tensor(inputs, img)
	interpreter.invoke()
	pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
	# boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
	boxes, pred_conf = pred[1], pred[0]

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
				

if __name__ == "__main__":

	cap = cv2.VideoCapture(0) #local camera
	while True:
		ret,frame=cap.read()
		if ret:
			image = preprocess(frame)
			pred_bbox = run_detection(image)
			image,results = draw_bbox(frame, pred_bbox)
			print(results)
			cv2.imshow('frame',image)
			if cv2.waitKey(1)==ord('q'):
				break