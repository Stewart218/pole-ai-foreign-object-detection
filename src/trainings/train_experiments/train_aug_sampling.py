"""
============================================================
Week3 Task2

Experiment 3

Hard Sample Augmentation
and Sampling Optimization


实验目的:

针对Baseline模型:
    小目标漏检
    遮挡目标检测困难
    复杂背景误检

进行数据层优化。


优化策略:

1. 强化数据增强
2. 增加样本变化
3. 提升困难场景泛化能力


模型:
    YOLOv8n


输出:
    runs/train/week3_aug_sampling


============================================================
"""


from pathlib import Path
import logging

from ultralytics import YOLO



# ==========================================================
# 路径
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
# 日志
# ==========================================================


logging.basicConfig(
    level=logging.INFO,
    format=
    "%(asctime)s - %(levelname)s - %(message)s"
)



# ==========================================================
# Trainer
# ==========================================================


class AugSamplingTrainer:


    def __init__(self):

        self.model_name = (
            "yolov8n.pt"
        )


        self.exp_name = (
            "week3_aug_sampling"
        )


        self.model = None



    # ------------------------------------------------------
    # Dataset检查
    # ------------------------------------------------------

    def check_dataset(self):

        if not DATASET_YAML.exists():

            raise FileNotFoundError(
                f"""
Dataset不存在:

{DATASET_YAML}
"""
            )


        logging.info(
            "Dataset:"
        )

        logging.info(
            str(DATASET_YAML)
        )



    # ------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------

    def load_model(self):

        logging.info(
            "Loading YOLOv8n..."
        )


        self.model = YOLO(
            self.model_name
        )



    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    def train(self):


        logging.info(
            "="*60
        )


        logging.info(
            "Week3 Task2"
        )


        logging.info(
            "Experiment 3"
        )


        logging.info(
            "Hard Sample Augmentation"
        )


        logging.info(
            "="*60
        )



        results = self.model.train(


            # ======================
            # Dataset
            # ======================

            data=str(
                DATASET_YAML
            ),



            # ======================
            # 基础参数
            # ======================

            epochs=70,

            imgsz=640,

            batch=16,



            # ======================
            # 优化器
            # 保持Baseline一致
            # ======================

            optimizer="auto",

            lr0=0.01,



            # ======================
            # 强化增强
            # ======================


            mosaic=1.0,


            mixup=0.15,


            copy_paste=0.2,


            degrees=10,


            translate=0.2,


            scale=0.8,


            shear=2,


            perspective=0.0005,



            hsv_h=0.02,

            hsv_s=0.8,

            hsv_v=0.5,



            fliplr=0.5,



            # ======================
            # GPU
            # ======================

            device=0,


            workers=8,



            # ======================
            # 保存
            # ======================

            project=str(
                RUN_DIR
            ),


            name=self.exp_name,


            save=True,


            plots=True,



            # ======================
            # 固定随机种子
            # ======================

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
        "="*60
    )

    print(
        "Week3 Task2"
    )

    print(
        "Experiment 3"
    )

    print(
        "Hard Sample Augmentation"
    )

    print(
        "="*60
    )


    trainer = AugSamplingTrainer()


    trainer.check_dataset()


    trainer.load_model()


    trainer.train()



    print()

    print(
        "Training Finished!"
    )


    print(
        "Results:"
    )


    print(
        RUN_DIR /
        trainer.exp_name
    )



if __name__ == "__main__":

    main()