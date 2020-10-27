import cv2
import tensorflow as tf
import colorsys
import random
import numpy as np
from collections import defaultdict



def non_max_suppression(boxes,scores,iou_threshold=0.45,score_threshold=0.2):
    selected_indices = []
    classes = np.argmax(scores, axis=1)
    scores = np.amax(scores,axis = 1)
    indices = np.where(scores>=score_threshold)
    scores = scores[indices]
    boxes = boxes[indices]
    classes = classes[indices]

    indices = scores.argsort()[::-1]
    areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)

    while indices.size>0:
        i = indices[0]
        selected_indices.append(i)

        ixy0 = np.maximum(boxes[indices[0],:2],boxes[indices[1:],:2])
        ixy1 = np.minimum(boxes[indices[0],2:],boxes[indices[1:],2:])

        inter = np.prod(ixy1-ixy0,axis=1)*(ixy1>ixy0).all(axis=1)
        ious = inter/(areas[indices[1:]]+areas[i]-inter)

        selected = np.where(ious<iou_threshold)[0] + 1
        indices = indices[selected]
    
    return boxes[selected_indices], scores[selected_indices], classes[selected_indices]





def preprocess(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416))
    image = np.array([image]).astype(np.float32)
    image = image/255.
    return image
def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    # method1
    # mask = scores_max >= score_threshold
    # class_boxes = tf.boolean_mask(box_xywh, mask)
    # pred_conf = tf.boolean_mask(scores, mask)

    # method2
    mask = tf.where(scores_max>=score_threshold)
    class_boxes = tf.gather_nd(box_xywh, mask)
    pred_conf = tf.gather_nd(scores, mask)

    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return (boxes, pred_conf)


def read_class_names():
    names ={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    return names







def draw_bbox(image, bboxes, classes=read_class_names(), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    color_green=(0,255,0)
    color_red=(0,0,255)
    # random.seed(0)
    # random.shuffle(colors)
    # random.seed(None)
    out_boxes, out_scores, out_classes, num_boxes = bboxes

    results_dic = defaultdict(list)

    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])

        results_dic[classes[class_ind]].append(coor.tolist())
        if classes[class_ind] == 'knife' or classes[class_ind] == 'scissors':
            bbox_color = color_red
        else:
            bbox_color = color_green
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image, results_dic


def draw_bbox_new(image, bboxes, classes=read_class_names(), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    color_green=(0,255,0)
    color_red=(0,0,255)
    # random.seed(0)
    # random.shuffle(colors)
    # random.seed(None)
    out_boxes, out_scores, out_classes, num_boxes = bboxes

    results_dic = defaultdict(list)

    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        score = out_scores[i]
        if score<0.2:
            break
        coor = out_boxes[i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        fontScale = 0.5
        
        class_ind = int(out_classes[i])

        results_dic[classes[class_ind]].append(coor.tolist())
        if classes[class_ind] == 'knife' or classes[class_ind] == 'scissors':
            bbox_color = color_red
        else:
            bbox_color = color_green
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])

        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image, results_dic