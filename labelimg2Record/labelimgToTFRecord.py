import tensorflow as tf  # 导入TensorFlow
import cv2  # 导入OpenCV
import os  # 用于文件操作
import glob  # 用于遍历文件夹内的xml文件
import xml.etree.ElementTree as ET  # 用于解析xml文件
from PIL import Image
from object_detection.utils import dataset_util

# 将LabelImg标注的图像文件和标注信息保存为TFRecord
class LabelImg2TFRecord:

    @classmethod
    def gen(cls, path):
        """
        :param path: LabelImg标识文件的路径，及生成的TFRecord文件路径
        """
        # 遍历文件夹内的全部xml文件，1个xml文件描述1个图像文件的标注信息
        writer = tf.python_io.TFRecordWriter(os.path.join(path, "train.record"))
        for f in glob.glob(path + "/*.xml"):
            # 解析xml文件
            try:
                tree = ET.parse(f)
            except FileNotFoundError:
                print("无法找到xml文件: "+f)
                return False
            except ET.ParseError:
                print("无法解析xml文件: "+f)
                return False
            else:  # ET.parse()正确运行

                # 取得xml根节点
                root = tree.getroot()

                # 取得图像路径和文件名
                img_name = root.find("filename").text
                img_path = root.find("path").text

                # 取得图像宽高
                img_width = int(root.find("size")[0].text)
                img_height = int(root.find("size")[1].text)

                # 取得所有标注object的信息
                label = []  # 类别名称
                xmin = []
                xmax = []
                ymin = []
                ymax = []
                classes = []
                image_format = b'jpg'
                # 查找根节点下全部名为object的节点
                for m in root.findall("object"):
                    xmin.append(float(m[4][0].text) / img_width)
                    xmax.append(float(m[4][2].text) / img_width)
                    ymin.append(float(m[4][1].text) / img_height)
                    ymax.append(float(m[4][3].text) / img_height)
                    # 用encode将str类型转为bytes类型，相应的用decode由bytes转回str类型
                    label.append(m[0].text.encode("utf-8"))
                    classes.append(class_text_to_int(m[0].text))

                print(label)
                print(classes)
                # 至少有1个标注目标
                if len(label) > 0:
                    print(" ========================== ")

                    with tf.gfile.GFile(os.path.join(path, '{}'.format(img_name)), 'rb') as fid:
                        encoded_jpg = fid.read()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image/height': dataset_util.int64_feature(img_height),
                        'image/width': dataset_util.int64_feature(img_width),
                        'image/filename': dataset_util.bytes_feature(img_name.encode("utf-8")),
                        'image/source_id': dataset_util.bytes_feature(img_name.encode("utf-8")),
                        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                        'image/format': dataset_util.bytes_feature(image_format),
                        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                        'image/object/class/text': dataset_util.bytes_list_feature(label),
                        'image/object/class/label': dataset_util.int64_list_feature(classes),
                    }))
                    # 数据写入TFRecord文件
                    writer.write(example.SerializeToString())
                else:
                    print("xml文件{0}无标注目标".format(f))
                    return False

        print("完成全部xml标注文件的保存")
        return True

def class_text_to_int(row_label):
    if row_label == 'up':
        return 1
    if row_label == 'leftClick':
        return 2
    if row_label == 'move':
        return 3
    else:
        None

if __name__ == "__main__":
    LabelImg2TFRecord.gen("F:/python/testdata")

