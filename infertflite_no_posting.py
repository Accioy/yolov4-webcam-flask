import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import cv2
from utils import preprocess,draw_bbox,filter_boxes,draw_bbox_new
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession



def load_model_lite(path):
    #input_size=416
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details[0]['index'], output_details



interpreter,inputs,output_details = load_model_lite('model/yolov4-416-chg1.tflite')
print('input_details:',inputs)
print('output_details:',output_details)
print(len(output_details))



def run_detection(img):
    global interpreter,inputs,output_details
    input_size=416
    interpreter.set_tensor(inputs, img)
    
    try:
        print('invoke')
        interpreter.invoke()
        print('invoke fin')
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    except:
        print('except')
        return [0,0,0,0]
    
    valid_detections=pred[0].shape[0]
    #pred_bbox = [boxes.numpy(), selected_scores.numpy(), classes.numpy(), valid_detections]
    pred_bbox = [pred[1],pred[2], pred[0], valid_detections]
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