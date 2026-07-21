"""
============================================================
Week3 Task2

Experiment 1

YOLOv8s Model Comparison

实验目的:
    在Baseline YOLOv8n基础上，
    更换更高容量模型YOLOv8s，
    分析模型规模提升对检测性能的影响。

对比:

Baseline:
    YOLOv8n

Experiment:
    YOLOv8s


保持一致参数:
    epochs=70
    imgsz=640
    batch=16
    optimizer=auto
    数据增强保持一致


输出:
    runs/train/week3_yolov8s/

============================================================
"""


from pathlib import Path
import logging

from ultralytics import YOLO



# ==========================================================
# 项目路径配置
# ==========================================================


# 当前文件:
# E:\Pole_AI_Project\src\trainings\train_yolov8s.py


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)



# 数据集配置文件

DATASET_YAML = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "dataset.yaml"
)



# 训练结果保存目录

RUN_DIR = (
    PROJECT_ROOT
    /
    "runs"
    /
    "train"
)



# ==========================================================
# 日志
# ==========================================================


logging.basicConfig(
    level=logging.INFO,
    format=
    "%(asctime)s - %(levelname)s - %(message)s"
)



# ==========================================================
# YOLOv8s训练器
# ==========================================================


class YOLOv8STrainer:
    """
    YOLOv8s模型训练实验

    用于Week3 Task2:
    Experiment 1
    """


    def __init__(self):

        # 使用YOLOv8 small模型

        self.model_name = (
            "yolov8s.pt"
        )


        self.exp_name = (
            "week3_yolov8s"
        )


        self.model = None



    # ------------------------------------------------------
    # 检查数据集
    # ------------------------------------------------------

    def check_dataset(self):

        if not DATASET_YAML.exists():

            raise FileNotFoundError(
                f"""
Dataset yaml不存在:

{DATASET_YAML}

请检查:
1. 是否完成prepare_dataset.py
2. dataset.yaml是否位于data/processed/
"""
            )


        logging.info(
            "Dataset found:"
        )

        logging.info(
            str(DATASET_YAML)
        )



    # ------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------

    def load_model(self):

        logging.info(
            "Loading YOLOv8s model..."
        )


        self.model = YOLO(
            self.model_name
        )


    # ------------------------------------------------------
    # 模型训练
    # ------------------------------------------------------

    def train(self):


        logging.info(
            "=" * 60
        )

        logging.info(
            "Week3 Task2"
        )

        logging.info(
            "Experiment 1"
        )

        logging.info(
            "YOLOv8s Baseline Comparison"
        )

        logging.info(
            "=" * 60
        )



        results = self.model.train(


            # ==========================
            # 数据集
            # ==========================

            data=str(
                DATASET_YAML
            ),



            # ==========================
            # 基础参数
            # ==========================

            epochs=70,

            imgsz=640,

            batch=16,



            # ==========================
            # 优化器
            # 与Baseline保持一致
            # ==========================

            optimizer="auto",

            lr0=0.01,

            lrf=0.01,


            momentum=0.937,


            weight_decay=0.0005,



            # ==========================
            # 数据增强
            # 保持一致便于公平比较
            # ==========================

            mosaic=1.0,

            mixup=0.0,


            hsv_h=0.015,

            hsv_s=0.7,

            hsv_v=0.4,



            # ==========================
            # GPU
            # ==========================

            device=0,

            workers=8,



            # ==========================
            # 保存
            # ==========================

            project=str(
                RUN_DIR
            ),

            name=self.exp_name,


            save=True,

            plots=True,



            # ==========================
            # 复现
            # ==========================

            seed=42,

            deterministic=True,


            pretrained=True

        )


        return results




# ==========================================================
# main
# ==========================================================


def main():


    print()

    print(
        "=" * 60
    )

    print(
        "Week3 Task2"
    )

    print(
        "Experiment 1"
    )

    print(
        "YOLOv8s Model Comparison"
    )

    print(
        "=" * 60
    )



    trainer = YOLOv8STrainer()


    trainer.check_dataset()


    trainer.load_model()


    trainer.train()



    print()

    print(
        "Training Finished!"
    )


    print(
        "Results saved:"
    )


    print(
        RUN_DIR /
        trainer.exp_name
    )



if __name__ == "__main__":

    main()