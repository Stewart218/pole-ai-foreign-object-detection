"""
============================================================

train_aug_v2.py

Week3 Task2 / Task4

Experiment 4:

YOLOv8n
+
Class Balanced Data Augmentation


输入:
data/processed_augmented/dataset.yaml


输出:

runs/train/
    week3_aug_v2/

        weights/
            best.pt
            last.pt

        results.csv

        results.png

        confusion_matrix.png

        PR_curve.png


============================================================
"""


from ultralytics import YOLO

from pathlib import Path

import torch



# ============================================================
# Path Configuration
# ============================================================

from pathlib import Path


# 自动获取项目根目录
# train_aug_v2.py位置:
# src/trainings/train_experiments/train_aug_v2.py

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)



# Final Dataset
# 最终数据集

DATA_YAML = (
    PROJECT_ROOT
    /
    "data"
    /
    "final_dataset"
    /
    "dataset.yaml"
)



# pretrained model

MODEL = "yolov8n.pt"



# Training output directory

SAVE_DIR = (
    PROJECT_ROOT
    /
    "runs"
    /
    "train"
)



# Experiment name

EXPERIMENT_NAME = (
    "week3_aug_v2"
)





# ============================================================
# Trainer
# ============================================================


class AugmentedTrainer:


    def __init__(self):


        self.model = YOLO(
            MODEL
        )



    def check_environment(self):


        print("="*60)

        print(
            "Experiment 4"
        )

        print(
            "YOLOv8n + Class Balanced Augmentation"
        )

        print("="*60)



        print(

            "Dataset:",

            DATA_YAML

        )


        print(

            "CUDA:",

            torch.cuda.is_available()

        )


        if torch.cuda.is_available():

            print(

                torch.cuda.get_device_name(0)

            )



        print("="*60)





    def train(self):


        self.check_environment()



        results = self.model.train(



            # -------------------------
            # Dataset
            # -------------------------

            data=str(
                DATA_YAML
            ),



            # -------------------------
            # Training
            # -------------------------

            epochs=100,


            imgsz=640,


            batch=16,



            device=0,



            workers=2,



            cache=False,



            pretrained=True,



            seed=42,



            deterministic=True,



            # -------------------------
            # Optimizer
            # -------------------------


            optimizer="AdamW",



            lr0=0.001,



            lrf=0.01,



            momentum=0.937,



            weight_decay=0.0005,



            # -------------------------
            # Augmentation
            # -------------------------

            hsv_h=0.015,


            hsv_s=0.7,


            hsv_v=0.4,


            degrees=5,


            translate=0.1,


            scale=0.5,


            shear=2,


            fliplr=0.5,


            flipud=0.0,



            mosaic=1.0,


            close_mosaic=10,



            mixup=0.0,



            copy_paste=0.0,



            # -------------------------
            # Validation
            # -------------------------


            val=True,



            plots=True,



            # -------------------------
            # Save
            # -------------------------


            project=str(

                SAVE_DIR

            ),



            name=EXPERIMENT_NAME,



            exist_ok=True,


        )



        return results





# ============================================================
# Main
# ============================================================


def main():


    trainer = AugmentedTrainer()


    trainer.train()



    print("\n")

    print("="*60)

    print(

        "Experiment 4 Finished!"

    )

    print("="*60)



    print(

        "Weights saved at:"

    )


    print(

        SAVE_DIR

        /

        EXPERIMENT_NAME

        /

        "weights"

    )





if __name__ == "__main__":

    main()