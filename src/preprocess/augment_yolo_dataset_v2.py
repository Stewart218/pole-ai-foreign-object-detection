"""
============================================================
YOLO Dataset Class-aware Augmentation V2

功能:
1. 读取YOLO格式训练集
2. 根据label自动识别类别
3. 针对不同类别执行不同增强
4. 同步生成image和label

输入:
data/annotation/images/train
data/annotation/labels/train

输出:
data/yolo_augmented/images/train
data/yolo_augmented/labels/train

============================================================
"""

import cv2
import random
import shutil
import logging

from pathlib import Path
from tqdm import tqdm

import numpy as np


# ==========================================================
# 路径配置
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


INPUT_IMAGE_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "annotation"
    /
    "images"
    /
    "train"
)


INPUT_LABEL_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "annotation"
    /
    "labels"
    /
    "train"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "yolo_augmented"
)


OUTPUT_IMAGE_DIR = (
    OUTPUT_DIR
    /
    "images"
    /
    "train"
)


OUTPUT_LABEL_DIR = (
    OUTPUT_DIR
    /
    "labels"
    /
    "train"
)



# ==========================================================
# 类别配置
# ==========================================================

CLASS_NAMES = {

    0: "bird_nest",

    1: "balloon",

    2: "plastic_bag",

    3: "other_foreign_object"

}



# ==========================================================
# 增强次数配置
# ==========================================================

AUG_TIMES = {


    # 鸟巢已经效果很好
    "bird_nest":
        1,


    # 气球增加颜色变化
    "balloon":
        3,


    # 塑料袋重点增强
    "plastic_bag":
        4,


    # 其他异物
    "other_foreign_object":
        3

}



# ==========================================================
# 日志
# ==========================================================


logging.basicConfig(
    level=logging.INFO,
    format=
    "%(asctime)s - %(levelname)s - %(message)s"
)



# ==========================================================
# YOLO标签读取
# ==========================================================


def read_yolo_label(label_path):

    """
    读取YOLO txt

    返回:
    [
      [class,x,y,w,h],
      ...
    ]

    """

    boxes = []


    if not label_path.exists():
        return boxes


    with open(label_path,"r") as f:

        lines=f.readlines()


    for line in lines:

        values=line.strip().split()


        if len(values)==5:

            boxes.append(
                [
                    int(values[0]),
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4])
                ]
            )


    return boxes



def get_main_class(label_path):

    """
    根据YOLO标签判断类别

    默认取第一目标类别

    """

    boxes = read_yolo_label(label_path)


    if len(boxes)==0:

        return None


    class_id = boxes[0][0]


    return CLASS_NAMES[class_id]



# ==========================================================
# 创建输出目录
# ==========================================================


def create_output_dirs():


    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    OUTPUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# Block 2
# Image Enhancement Functions
#
# 功能：
# 1. Brightness enhancement
# 2. HSV color augmentation
# 3. Random crop with bbox synchronization
#
# 输入：
# image:
#     OpenCV读取的BGR图片
#
# labels:
#     YOLO格式标签
#     [
#       [class_id, x_center, y_center, w, h],
#       ...
#     ]
#
# 输出：
# image_aug
# labels_aug
#
# ============================================================


import cv2
import random
import numpy as np


# ============================================================
# 1. Brightness Enhancement
# ============================================================

def augment_brightness(
        image,
        factor_range=(0.7, 1.4)
):
    """
    亮度增强

    参数:
        image:
            BGR图片

        factor_range:
            亮度倍率范围


    返回:
        enhanced_image

    """

    factor = random.uniform(
        factor_range[0],
        factor_range[1]
    )


    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    hsv = hsv.astype(np.float32)


    # 调整V通道
    hsv[:, :, 2] *= factor


    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2],
        0,
        255
    )


    hsv = hsv.astype(
        np.uint8
    )


    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    return result



# ============================================================
# 2. HSV Color Augmentation
# ============================================================

def augment_hsv(
        image,
        h_gain=0.015,
        s_gain=0.7,
        v_gain=0.4
):
    """
    HSV颜色扰动

    模拟：

    - 不同天气
    - 光照变化
    - 摄像头颜色偏差


    与YOLO默认增强策略保持一致


    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    hsv = hsv.astype(
        np.float32
    )


    # Hue
    h_factor = random.uniform(
        -h_gain,
        h_gain
    )


    hsv[:, :, 0] += (
        hsv[:, :, 0]
        *
        h_factor
    )


    # Saturation

    s_factor = random.uniform(
        1-s_gain,
        1+s_gain
    )


    hsv[:, :, 1] *= s_factor



    # Value

    v_factor = random.uniform(
        1-v_gain,
        1+v_gain
    )


    hsv[:, :, 2] *= v_factor



    hsv[:, :, 0] = np.clip(
        hsv[:, :, 0],
        0,
        179
    )


    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255
    )


    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2],
        0,
        255
    )



    hsv = hsv.astype(
        np.uint8
    )


    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    return result



# ============================================================
# YOLO bbox转换工具
# ============================================================


def yolo_to_pixel(
        label,
        img_width,
        img_height
):
    """
    YOLO格式转换为像素坐标


    输入:

    label:
        [
        class_id,
        x_center,
        y_center,
        width,
        height
        ]

    输出:

    [
    class_id,
    x1,
    y1,
    x2,
    y2
    ]

    """


    cls, xc, yc, w, h = label


    x_center = xc * img_width
    y_center = yc * img_height


    box_width = w * img_width
    box_height = h * img_height



    x1 = (
        x_center
        -
        box_width / 2
    )


    y1 = (
        y_center
        -
        box_height / 2
    )


    x2 = (
        x_center
        +
        box_width / 2
    )


    y2 = (
        y_center
        +
        box_height / 2
    )


    return [
        int(cls),
        x1,
        y1,
        x2,
        y2
    ]




def pixel_to_yolo(
        bbox,
        img_width,
        img_height
):
    """
    像素bbox转YOLO格式
    """


    cls, x1, y1, x2, y2 = bbox



    x_center = (
        (x1+x2)/2
        /
        img_width
    )


    y_center = (
        (y1+y2)/2
        /
        img_height
    )


    width = (
        x2-x1
    ) / img_width


    height = (
        y2-y1
    ) / img_height



    return [
        cls,
        x_center,
        y_center,
        width,
        height
    ]



# ============================================================
# 3. Random Crop + bbox Synchronization
# ============================================================


def augment_random_crop(
        image,
        labels,
        crop_ratio=(0.7,0.95)
):
    """
    随机裁剪增强


    重点:

    裁剪后同步修改bbox


    参数:

    image:
        BGR图片


    labels:

        [
        class_id,
        xc,
        yc,
        w,
        h
        ]



    返回:

        crop_image

        crop_labels

    """


    h,w = image.shape[:2]


    crop_scale = random.uniform(
        crop_ratio[0],
        crop_ratio[1]
    )


    crop_w = int(
        w * crop_scale
    )


    crop_h = int(
        h * crop_scale
    )


    if crop_w >= w or crop_h >= h:

        return image, labels



    # 随机裁剪位置

    x_start = random.randint(
        0,
        w-crop_w
    )


    y_start = random.randint(
        0,
        h-crop_h
    )



    crop_img = image[
        y_start:y_start+crop_h,
        x_start:x_start+crop_w
    ]



    new_labels=[]



    for label in labels:


        bbox = yolo_to_pixel(
            label,
            w,
            h
        )


        cls,x1,y1,x2,y2 = bbox



        # bbox移动

        x1 -= x_start
        x2 -= x_start

        y1 -= y_start
        y2 -= y_start



        # 与裁剪区域求交

        x1=max(
            0,
            x1
        )

        y1=max(
            0,
            y1
        )


        x2=min(
            crop_w,
            x2
        )


        y2=min(
            crop_h,
            y2
        )



        # 判断目标是否还存在

        bw=x2-x1
        bh=y2-y1



        if bw <= 5 or bh <=5:

            continue



        new_bbox=[
            cls,
            x1,
            y1,
            x2,
            y2
        ]


        new_label=pixel_to_yolo(
            new_bbox,
            crop_w,
            crop_h
        )


        new_labels.append(
            new_label
        )



    return (
        crop_img,
        new_labels
    )
# ============================================================
# Block 3
# Dataset Loading
# Class-aware Augmentation
# Saving
# Report Generation
#
# ============================================================


from pathlib import Path
import cv2
import shutil
from collections import defaultdict



# ============================================================
# Dataset Configuration
# ============================================================


class AugmentConfig:
    """
    数据增强配置
    """


    # 原始数据

    source_image_dir = Path(
        r"/data/annotation\images\train"
    )


    source_label_dir = Path(
        r"/data/annotation\labels\train"
    )



    # 输出目录

    output_image_dir = Path(
        r"/data/augmented_dataset/images/train"
    )


    output_label_dir = Path(
        r"/data/augmented_dataset/labels/train"
    )



    # 类别名称

    class_names = {

        0:"bird_nest",

        1:"balloon",

        2:"plastic_bag",

        3:"other_foreign_object"

    }



    # 不同类别增强次数

    augment_times = {


        # 原始图片保留

        "bird_nest":2,


        # 漏检较多

        "balloon":4,


        # 最困难类别

        "plastic_bag":5,


        "other_foreign_object":4

    }




# ============================================================
# Label Reader
# ============================================================


def read_yolo_label(label_path):

    """
    读取YOLO标签


    返回:

    [
        [
        cls,
        xc,
        yc,
        w,
        h
        ],

        ...

    ]

    """


    labels=[]


    if not label_path.exists():

        return labels



    with open(
        label_path,
        "r"
    ) as f:


        lines=f.readlines()



    for line in lines:


        data=line.strip().split()



        if len(data)!=5:

            continue



        labels.append(

            [
                int(data[0]),

                float(data[1]),

                float(data[2]),

                float(data[3]),

                float(data[4])

            ]

        )


    return labels




# ============================================================
# Label Writer
# ============================================================


def save_yolo_label(
        label_path,
        labels
):


    with open(
        label_path,
        "w"
    ) as f:


        for label in labels:


            f.write(

                "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(

                    label[0],
                    label[1],
                    label[2],
                    label[3],
                    label[4]

                )

            )




# ============================================================
# Class Analyzer
# ============================================================


def analyze_class(labels):

    """
    根据label统计图片类别

    """

    classes=set()


    for label in labels:

        classes.add(
            label[0]
        )


    return classes




# ============================================================
# Select Augmentation Method
# ============================================================


def apply_random_augmentation(
        image,
        labels
):

    """
    随机选择增强方式

    """



    method=random.choice(

        [

            "brightness",

            "hsv",

            "crop"

        ]

    )



    if method=="brightness":


        return (

            augment_brightness(image),

            labels,

            method

        )



    elif method=="hsv":


        return (

            augment_hsv(image),

            labels,

            method

        )



    else:


        img,lab = augment_random_crop(

            image,

            labels

        )


        return (

            img,

            lab,

            method

        )





# ============================================================
# Main Dataset Augmentation
# ============================================================


def augment_dataset():


    config=AugmentConfig()



    # 创建目录


    config.output_image_dir.mkdir(

        parents=True,

        exist_ok=True

    )


    config.output_label_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    statistics=defaultdict(int)



    image_files=list(

        config.source_image_dir.glob(

            "*.png"

        )

    )



    print("="*60)

    print(
        "Start Dataset Augmentation"
    )

    print(
        f"Original images: {len(image_files)}"
    )

    print("="*60)




    for image_path in image_files:



        label_path=(

            config.source_label_dir

            /

            (

                image_path.stem

                +

                ".txt"

            )

        )



        labels=read_yolo_label(

            label_path

        )



        if len(labels)==0:

            continue



        classes=analyze_class(

            labels

        )



        # 取主要类别

        cls_id=list(classes)[0]



        cls_name=config.class_names[cls_id]



        times=config.augment_times[cls_name]




        # ======================
        # 保存原图
        # ======================


        shutil.copy(

            image_path,

            config.output_image_dir

            /

            image_path.name

        )


        shutil.copy(

            label_path,

            config.output_label_dir

            /

            label_path.name

        )



        statistics[cls_name]+=1




        # ======================
        # 增强
        # ======================


        for i in range(times):



            image=cv2.imread(

                str(image_path)

            )



            aug_img,aug_labels,method = apply_random_augmentation(

                image,

                labels

            )



            new_name=(

                image_path.stem

                +

                f"_aug_{i}_{method}.png"

            )



            new_label=(

                image_path.stem

                +

                f"_aug_{i}_{method}.txt"

            )



            cv2.imwrite(

                str(

                    config.output_image_dir

                    /

                    new_name

                ),

                aug_img

            )



            save_yolo_label(

                config.output_label_dir

                /

                new_label,

                aug_labels

            )


            statistics[cls_name]+=1




    generate_report(

        statistics

    )



    print(
        "Augmentation Finished!"
    )




# ============================================================
# Report
# ============================================================


def generate_report(statistics):


    report_path=Path(

        r"/data/augmented_dataset/augmentation_report.txt"

    )


    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as f:



        f.write(

            "Dataset Augmentation Report\n"

        )


        f.write(

            "="*40+"\n\n"

        )



        total=0


        for cls,count in statistics.items():


            f.write(

                f"{cls}: {count}\n"

            )


            total+=count



        f.write(

            "\nTotal images: "

            +

            str(total)

        )



    print(

        f"Report saved: {report_path}"

    )





# ============================================================
# Entry
# ============================================================


if __name__=="__main__":


    augment_dataset()