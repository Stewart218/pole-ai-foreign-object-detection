"""
============================================================
Week3 Task2
Experiment 2

Learning Rate Optimization

实验目的:
    在Baseline YOLOv8n基础上，
    优化学习率策略与优化器参数，
    分析对检测性能的影响。

优化内容:
    1. SGD -> AdamW
    2. lr0: 0.01 -> 0.001
    3. cosine learning rate decay

输出:
    runs/train/week3_lr_opt/

============================================================
"""


from pathlib import Path
import logging
import sys

from ultralytics import YOLO



# ==========================================================
# 项目路径
# ==========================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


DATASET_YAML = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "dataset.yaml"
)


RUN_DIR = (
    PROJECT_ROOT
    /
    "runs"
    /
    "train"
)



# ==========================================================
# 日志配置
# ==========================================================


logging.basicConfig(
    level=logging.INFO,
    format=
    "%(asctime)s - %(levelname)s - %(message)s"
)



# ==========================================================
# 学习率优化训练器
# ==========================================================


class LROptimizationTrainer:
    """
    YOLOv8学习率优化实验
    """

    def __init__(self):

        self.model_name = "yolov8n.pt"

        self.exp_name = (
            "week3_lr_opt"
        )


    # ------------------------------------------------------
    # 数据检查
    # ------------------------------------------------------

    def check_dataset(self):

        if not DATASET_YAML.exists():

            raise FileNotFoundError(
                f"""
Dataset yaml not found:

{DATASET_YAML}

请检查:
1. 是否已经运行 prepare_dataset.py
2. dataset.yaml是否位于 data/processed/
"""
            )


        logging.info(
            "Dataset found:"
        )

        logging.info(
            str(DATASET_YAML)
        )


    # ------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------

    def load_model(self):

        logging.info(
            "Loading YOLOv8 model..."
        )


        self.model = YOLO(
            self.model_name
        )


    # ------------------------------------------------------
    # 开始训练
    # ------------------------------------------------------

    def train(self):


        logging.info(
            "="*60
        )

        logging.info(
            "Week3 Task2"
        )

        logging.info(
            "Experiment 2: Learning Rate Optimization"
        )

        logging.info(
            "="*60
        )


        results = self.model.train(

            # ----------------------
            # 数据
            # ----------------------

            data=str(
                DATASET_YAML
            ),


            # ----------------------
            # 基础训练参数
            # ----------------------

            epochs=70,

            imgsz=640,

            batch=16,


            # ----------------------
            # 优化器实验变量
            # ----------------------

            optimizer="AdamW",

            lr0=0.001,

            lrf=0.01,


            momentum=0.937,


            weight_decay=0.0005,


            # ----------------------
            # 学习率策略
            # ----------------------

            cos_lr=True,


            # ----------------------
            # 数据增强
            # 保持与Baseline一致
            # ----------------------

            mosaic=1.0,

            mixup=0.0,


            # ----------------------
            # 训练设备
            # ----------------------

            device=0,


            workers=8,


            # ----------------------
            # 保存
            # ----------------------

            project=str(
                RUN_DIR
            ),

            name=self.exp_name,


            save=True,

            plots=True,


            # ----------------------
            # 复现
            # ----------------------

            seed=42,


            deterministic=True,


            pretrained=True

        )


        return results



# ==========================================================
# Main
# ==========================================================


def main():


    print(
        "\n"
        "="*60
    )

    print(
        "Week3 Task2"
    )

    print(
        "Experiment 2"
    )

    print(
        "Learning Rate Optimization"
    )

    print(
        "="*60
    )


    trainer = (
        LROptimizationTrainer()
    )


    trainer.check_dataset()


    trainer.load_model()


    trainer.train()



    print(
        "\nTraining Finished!"
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