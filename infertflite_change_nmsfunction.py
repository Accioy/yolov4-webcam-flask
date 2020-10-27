import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import cv2
from utils import preprocess,draw_bbox_new,non_max_suppression
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
    try:
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    except:
        return [0,0,0,0]
    # boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    boxes, pred_conf = pred[1],pred[0]
    # scores_max = tf.math.reduce_max(pred_conf[0], axis=-1)
    # valid_indices,selected_scores = tf.image.non_max_suppression_with_scores(
    #     boxes=boxes[0],
    #     scores=scores_max,
    #     max_output_size=100,
    #     iou_threshold=0.45,
    #     score_threshold=0.25,
    #     soft_nms_sigma=0.0
    # )
    # boxes = tf.gather(boxes[0],valid_indices)
    # scores = tf.gather(pred_conf[0],valid_indices)
    # classes = tf.math.argmax(scores,1)
    #scores = tf.gather(scores_max,valid_indices)

    boxes, scores, classes = non_max_suppression(boxes[0],pred_conf[0])
    valid_detections=boxes.shape[0]
    pred_bbox = [boxes, scores, classes, valid_detections]
    return pred_bbox
                

if __name__ == "__main__":

    cap = cv2.VideoCapture(0) #local camera
    while True:
        ret,frame=cap.read()
        if ret:
            image = preprocess(frame)
            pred_bbox = run_detection(image)
            image,results = draw_bbox_new(frame, pred_bbox)
            print(results)
            cv2.imshow('frame',image)
            if cv2.waitKey(1)==ord('q'):
                break