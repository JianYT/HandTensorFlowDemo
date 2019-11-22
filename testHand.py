import time

import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import h5py
from object_detection.utils import visualization_utils as vis_util
# --------------Model preparation----------------
# Path to frozen detection graph. This is the actual model that is used for
# the object detection.
# PATH_TO_CKPT = 'F:/hand-gesture/object-detection/models/myConfig/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
# PATH_TO_LABELS = 'F:/hand-gesture/object-detection/models/research/object_detection/data/mscoco_label_map.pbtxt'

PATH_TO_CKPT = 'F:/hand-gesture/object-detection/models/research/object_detection/hand_train_log/save_model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'F:/python/data/train.pbtxt'

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

print('label',category_index)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular
# object was detected.
gboxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
gscores = detection_graph.get_tensor_by_name('detection_scores:0')
gclasses = detection_graph.get_tensor_by_name('detection_classes:0')
gnum_detections = detection_graph.get_tensor_by_name('num_detections:0')
camera = cv2.VideoCapture(0)
time.sleep(2)
# TODO: Add class names showing in the image
#frozen model
def detect_image_objects(image, sess, detection_graph):
    # Expand dimensions since the model expects images to have
    # shape: [1, None, None, 3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image, axis=0)

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [gboxes, gscores, gclasses, gnum_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    height, width = image.shape[:2]
    # print(boxes)
    for i in range(boxes.shape[0]):
        # print(' ==================== 11111111111 ',scores)
        if (scores is None or
                scores[i] > 0.5):
            # name = category_index[classes[0][i]]['name']
            print(' ==================== ',category_index[classes[0][i]]['name'])
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            xmin = int(xmin * width)
            xmax = int(xmax * width)

            score = None if scores is None else scores[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_x = np.max((0, xmin - 10))
            text_y = np.max((0, ymin - 10))
            cv2.putText(image, '  Detection score: ' + str(score),
                        (text_x, text_y), font, 0.4, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                          (0, 255, 0), 2)
    return image

# saved model
def detect_image_objects_saved(image,predict_fn):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = np.expand_dims(img, 0)
    # cv2.imshow('Frame12', image)
    # return img_rgb
    output_data = predict_fn({"images": img_rgb})

    scores = output_data['detection_scores']
    boxes = output_data['detection_boxes']
    classes = output_data['detection_classes']
    num_detections = len(boxes)
    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    # num_detections = int(num_detections[0])

    boxes_pixels = []
    for i in range(num_detections):
        # scale box to image coordinates
        box = boxes[i] * np.array([img.shape[0],
                                   img.shape[1], img.shape[0], img.shape[1]])
        box = np.round(box).astype(int)
        boxes_pixels.append(box)
    boxes_pixels = np.array(boxes_pixels)

    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.5, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]),
                      (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale,
                    (255, 255, 255), thickness)

    # for i in pick:
    for i in range(num_detections):
        if scores[i] > 0.05:
            box = boxes_pixels[i]
            box = np.round(box).astype(int)
            # Draw bounding box.
            image = cv2.rectangle(
                img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            label = category_index[classes[i]]['name']

            # Draw label (class index and probability).
            draw_label(image, (box[1], box[0]), label)

    return img






with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # meta_graph_def = tf.saved_model.loader.load(sess, ['serve'], "F:/hand-gesture/object-detection-react-master1/public/web_model/save_model")
        # signature = meta_graph_def.signature_def
        # predict_fn = tf.contrib.predictor.from_saved_model("F:/hand-gesture/object-detection-react-master1/public/web_model/save_model",signature_def_key="predict_object")
        # predict_fn = tf.contrib.predictor.from_saved_model("F:/hand-gesture/object-detection/models/research/object_detection/testLog/save_model/saved_model/")
        #tf.saved_model.loader.load(sess, ['serve'], 'F:/hand-gesture/object-detection/models/myConfig/ssd_mobilenet_v1_coco_2018_01_28/saved_model/')
        #print(test)
        while True:
            (ret, frame) = camera.read()

            if not ret:
                print('No Camera')
                break
            # image = detect_image_objects_saved(frame,predict_fn)

            image = detect_image_objects(frame, sess, detection_graph)
            cv2.imshow('Frame', image)
            if cv2.waitKey(5)&0xFF == 27:
                break
