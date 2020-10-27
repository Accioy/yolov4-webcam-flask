import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
import cv2
from utils import preprocess,draw_bbox
from tensorflow.python.saved_model import tag_constants
import threading
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



if __name__ == "__main__":
    saved_model_loaded = tf.saved_model.load('model/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    cap = cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        if ret:
            image = preprocess(frame)

            img = tf.constant(image)
            pred_bbox = infer(img)
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
            image = draw_bbox(frame, pred_bbox)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1)==ord('q'):
                break