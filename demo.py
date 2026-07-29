"""
====================================================
Transmission Line Foreign Object Detection Demo

输电线路异物智能检测系统

功能:
1. 图形化选择待检测图片
2. 自动加载最终YOLO模型
3. 异物目标检测
4. 边界框可视化
5. 输出类别与置信度
6. 保存检测结果

Model:
YOLOv8n + Class Balanced Augmentation

====================================================
"""


import sys
from pathlib import Path
import time

import cv2

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit
)

from PyQt6.QtGui import (
    QPixmap,
    QImage
)

from ultralytics import YOLO



# ==============================
# 路径
# ==============================


PROJECT_ROOT = Path(__file__).resolve().parent


MODEL_PATH = (
    PROJECT_ROOT
    /
    "models"
    /
    "best_model.pt"
)


RESULT_DIR = (
    PROJECT_ROOT
    /
    "demo_results"
)


RESULT_DIR.mkdir(
    exist_ok=True
)



# 类别

CLASS_NAMES = [
    "bird_nest",
    "balloon",
    "plastic_bag",
    "other_foreign_object"
]



# ==============================
# GUI
# ==============================


class DetectionGUI(QWidget):


    def __init__(self):

        super().__init__()


        self.setWindowTitle(
            "输电线路异物智能检测系统"
        )


        self.resize(
            1200,
            700
        )


        # 加载模型

        print("=" * 60)
        print("Loading final detection model...")
        print(MODEL_PATH)
        print("=" * 60)

        self.model = YOLO(
            str(MODEL_PATH)
        )

        print("Model loaded successfully!")


        self.image_path = None



        self.init_ui()



    def init_ui(self):


        # 图片显示

        self.original_label = QLabel(
            "原始图片"
        )

        self.result_label = QLabel(
            "检测结果"
        )


        self.original_label.setFixedSize(
            500,
            400
        )


        self.result_label.setFixedSize(
            500,
            400
        )


        # 按钮


        self.open_btn = QPushButton(
            "选择图片"
        )


        self.detect_btn = QPushButton(
            "开始检测"
        )


        self.save_btn = QPushButton(
            "保存结果"
        )


        self.open_btn.clicked.connect(
            self.open_image
        )


        self.detect_btn.clicked.connect(
            self.detect
        )


        self.save_btn.clicked.connect(
            self.save_result
        )



        # 文本框


        self.info_box = QTextEdit()

        self.info_box.setReadOnly(
            True
        )



        # 布局


        image_layout = QHBoxLayout()

        image_layout.addWidget(
            self.original_label
        )

        image_layout.addWidget(
            self.result_label
        )



        button_layout = QHBoxLayout()

        button_layout.addWidget(
            self.open_btn
        )

        button_layout.addWidget(
            self.detect_btn
        )

        button_layout.addWidget(
            self.save_btn
        )



        layout = QVBoxLayout()


        layout.addLayout(
            image_layout
        )


        layout.addLayout(
            button_layout
        )


        layout.addWidget(
            self.info_box
        )


        self.setLayout(
            layout
        )



    # ==========================
    # 打开图片
    # ==========================


    def open_image(self):


        file,_ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )


        if file:

            self.image_path = file


            pixmap = QPixmap(
                file
            )


            self.original_label.setPixmap(
                pixmap.scaled(
                    500,
                    400
                )
            )



    # ==========================
    # 检测
    # ==========================


    def detect(self):


        if self.image_path is None:

            self.info_box.append(
                "请先选择图片"
            )

            return



        start=time.time()



        result = self.model.predict(
            self.image_path,
            conf=0.25
        )[0]


        cost=time.time()-start



        # 绘制结果

        img=result.plot()

        save_path = (
                RESULT_DIR
                /
                f"result_{Path(self.image_path).name}"
        )


        cv2.imwrite(
            str(save_path),
            img
        )


        pixmap=QPixmap(
            str(save_path)
        )


        self.result_label.setPixmap(
            pixmap.scaled(
                500,
                400
            )
        )



        self.info_box.clear()


        self.info_box.append(
            f"推理时间: {cost*1000:.2f} ms\n"
        )


        if len(result.boxes)==0:

            self.info_box.append(
                "未检测到异物"
            )

        else:

            for box in result.boxes:


                cls=int(
                    box.cls[0]
                )


                conf=float(
                    box.conf[0]
                )


                self.info_box.append(
                    f"{CLASS_NAMES[cls]} : {conf:.3f}"
                )



    # ==========================
    # 保存
    # ==========================


    def save_result(self):


        self.info_box.append(
            "结果已保存至 demo_results/"
        )



# ==============================
# Main
# ==============================


if __name__=="__main__":


    app=QApplication(sys.argv)


    window=DetectionGUI()


    window.show()


    sys.exit(
        app.exec()
    )