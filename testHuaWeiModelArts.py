from datetime import datetime
import os
import time

import threading
import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import h5py

camera = cv2.VideoCapture(0)
time.sleep(2)
ClickStr = ""

screen_width, screen_height = pyautogui.size()
stopX = 0
stopY = 0
number = 0
stopNumber = 0

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

def detect_image_objects_saved_Huawei(image,predict_fn,labels_list):
    a = int(round(time.time() * 1000))
    image = cv2.flip(image, 1, dst=None)
    # 将图片转成tensorflow所使用的RGB 提高识别率
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h = img.shape[0]
    w = img.shape[1]
    WRatio = screen_width/w
    HRatio =  screen_height/h
    img_rgb = np.expand_dims(img, 0)

    # 传入rgb图片 进行识别
    output_data = predict_fn({"images": img_rgb})
    # 获取识别分数
    scores = output_data['detection_scores']
    # 获取识别位置
    boxes = output_data['detection_boxes']
    # 获取识别类型
    classes = output_data['detection_classes']
    num_detections = len(boxes)
    databoxes = []
    datascores = []
    datalabel = []
    # 遍历识别到的个数
    for i in range(num_detections):
        # 只针对分数大于某一个数的进行
        if scores[i] > 0.8:
            class_id = classes[i] - 1
            datalabel.append(labels_list[int(class_id)])
            databoxes.append(boxes[i])
            datascores.append(scores[i])

    if len(databoxes) == 0:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    bounding_boxes = databoxes
    confidence_score = datascores
    # Bounding boxes
    databoxes = np.array(bounding_boxes)
    picked_boxes = []
    picked_score = []
    picked_classes = []
    picked_datalabel = []
    if len(databoxes) != 0:
        # 分数转数组
        score = np.array(confidence_score)
        # 分数从大到小排列
        order = np.argsort(score)
        while order.size > 0:
            # 获取最大的的分数
            index = order[-1]
            # 只添加分数最高的一个识别对象
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])
            picked_classes.append(classes[index])
            picked_datalabel.append(datalabel[index])
            break
    for i in range(len(picked_boxes)):
        if picked_score[i] > 0.6:
            box = picked_boxes[i]
            box = np.round(box).astype(int)
            # 画边框
            image = cv2.rectangle(
                img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            # 画标注
            draw_label(image, (box[1], box[0]), picked_datalabel[i])
    if len(picked_boxes) > 0 :
        global ClickStr
        global number
        global stopX
        global stopY
        global stopNumber
        if picked_datalabel[0] == 'move':
            wa = np.array(picked_boxes)
            start_x = wa[:, 0]
            start_y = wa[:, 1]
            # end_x = wa[:, 2]
            # end_y = wa[:, 3]
            # centerX = (start_x + end_x)/2
            # centerY = (start_y + end_y)/2

            if ClickStr == "移动":

                y = start_y * WRatio * 1.5
                x = start_x * HRatio * 1.5
                if y > screen_width:
                    y = screen_width
                if x > screen_height:
                    x = screen_height
                if move(y,x):
                    # 移动超过一定范围五次之后才算退出其他状态
                    number += 1
                    print(number)
                    if number >= 3:
                        pyautogui.moveTo(y, x)
                        stopNumber = 0
                else:
                    stopNumber+=1
                    currentMouseX, currentMouseY = pyautogui.position()
                    stopX = currentMouseX
                    stopY = currentMouseY
                    # 五次的移动范围不超过一定值之后 则视为停止状态
                    if stopNumber >= 10:
                        ClickStr = "停止"
                        number = 0
                        print(ClickStr)

            ClickStr = "移动"
        if picked_datalabel[0] == 'up' and ClickStr != "左键":
            number = 0
            ClickStr = "左键"
            pyautogui.click(stopX,stopY,button='left')
        if picked_datalabel[0] == 'open' and ClickStr != "右键":
            number = 0
            ClickStr = "右键"
            pyautogui.click(stopX,stopY,button='right')
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 防止抖动
def move(centerY,centerX):
    currentMouseX, currentMouseY = pyautogui.position()
    x =  float(currentMouseX - centerY)
    y = float(currentMouseY - centerX)
    if (x < 3.5 and x > -3.5) or (y < 3.5 and y > -3.5):
        return False

    return True

with tf.Session() as sess:
    predict_fn = tf.contrib.predictor.from_saved_model(
        "F:/hand-gesture/object-detection-react-master1/public/web_model/save_model_small",
        signature_def_key="predict_object")
    h5f = h5py.File(os.path.join("F:/hand-gesture/object-detection-react-master1/public/web_model/save_model_small", 'index'), 'r')
    labels_list = h5f['labels_list'][:]
    # 遍历解码
    labels_list = [label.decode('utf-8') if isinstance(label, bytes) else label for label in labels_list]
    h5f.close()

    while True:
        (ret, frame) = camera.read()

        if not ret:
            print('No Camera')
            break
        image = detect_image_objects_saved_Huawei(frame, predict_fn,labels_list)
        cv2.imshow('Frame', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break