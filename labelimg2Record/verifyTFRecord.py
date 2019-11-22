import tensorflow as tf
import cv2
import numpy as np

# 获取文件列表



files = tf.train.match_filenames_once("F:/python/data/train.record")



# 创建输入队列
# 重复输出输入文件列表中的所有文件名，除非用参数num_epochs指定每个文件可轮询的次数
# shuffle参数用于控制随机打乱文件排序
# 返回值说明:
# A queue with the output strings. A QueueRunner for the Queue is added to the current Graph's QUEUE_RUNNER collection.
# 用tf.train.start_queue_runners启动queue的输出
queue = tf.train.string_input_producer(files, shuffle=True)

# 建立TFRecordReader并解析TFRecord文件
reader = tf.TFRecordReader()
_, serialized_example = reader.read(queue)  # tf.TFRecordReader.read()用于读取queue中的下一个文件
rec_features = tf.parse_single_example(  # 返回字典，字典key值即features参数中的key值
    serialized_example,
    features={
        "image/filename": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "image/width": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "image/height": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "image/encoded": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "image/object/class/text": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "image/object/class/label": tf.VarLenFeature(dtype=tf.int64),
        "image/object/bbox/xmin": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.VarLenFeature(dtype=tf.float32),
    }
)


# # 将tf.string转化成tf.uint8的tensor
# img_tensor = tf.decode_raw(rec_features["data"], tf.uint8)
# print(img_tensor.shape)

with tf.Session() as sess:
    """
    sess.run(tf.global_variables_initializer())
    print(sess.run(files))
    上述代码运行出错，提示如下：
    Attempting to use uninitialized value matching_filenames
    因为tf.train.match_filenames_once使用的是局部变量，非全局变量
    需要改成下方代码才能正确运行
    """
    sess.run(tf.local_variables_initializer())
    print(sess.run(files))  # 打印文件列表

    # 用子线程启动tf.train.string_input_producer生成的queue
    coord = tf.train.Coordinator()  # 用于控制线程结束
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 读出TFRecord文件内容
    for i in range(20):
        # 每次run都由string_input_producer更新至下一个TFRecord文件
        rec = sess.run(rec_features)
        print(rec["image/filename"].decode("utf-8"))  # 由bytes类型转为str类型
        print("目标数目: " + str(rec["image/object/class/label"].values.size))
        print(rec["image/object/class/label"].values)
        print(rec["image/object/bbox/xmin"].values)
        print(rec["image/object/bbox/xmax"].values)
        print(rec["image/object/bbox/ymin"].values)
        print(rec["image/object/bbox/ymax"].values)
        print(rec["image/height"])
        print(rec["image/width"])

        img = tf.image.decode_jpeg(rec['image/encoded'])
        # 调整图片大小
        img_resize = tf.image.resize_image_with_crop_or_pad(img, rec["image/height"].astype(np.int32), rec["image/width"].astype(np.int32))

        boxes = tf.constant([[[rec["image/object/bbox/ymin"].values[0],rec["image/object/bbox/xmin"].values[0], rec["image/object/bbox/ymax"].values[0],rec["image/object/bbox/xmax"].values[0]]]])
        # 给原始图片添加一个图层
        batched = tf.expand_dims(tf.image.convert_image_dtype(img_resize, tf.float32), 0)
        # 把boxes标注的框画到原始图片上
        image_with_boxes = tf.image.draw_bounding_boxes(batched, boxes)
        # 重新将原始图片设置为RGB
        image_with_boxes = tf.reshape(image_with_boxes, [rec["image/height"].astype(np.int32), rec["image/width"].astype(np.int32), 3])

        img_array = image_with_boxes.eval()
        ss =  rec["image/height"].astype(np.int32) * rec["image/object/bbox/xmin"].values[0]
        ss1 = rec["image/width"].astype(np.int32) * rec["image/object/bbox/ymin"].values[0]
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img = cv2.putText(img,
                          str(rec["image/object/class/label"].values[0])+" : "+str(rec["image/object/class/text"].decode("utf-8")),
                          (20,20),
                          cv2.FONT_HERSHEY_PLAIN , 1 , (0, 255, 0))

        img = cv2.putText(img,
                           "fileName : " + rec["image/filename"].decode("utf-8"),
                          (20, 50),
                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('123', img)
        cv2.waitKey()

    coord.request_stop()  # 结束线程
    coord.join(threads)  # 等待线程结束

