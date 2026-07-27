"""
============================================================
Week3 Task2

Experiment 3

YOLOv8n Small Object Enhancement


优化目标:

1. 提升小目标异物检测能力
2. 降低过强缩放造成的信息损失
3. 增强困难背景泛化能力


策略:

- YOLOv8n模型保持不变
- 提高输入分辨率 640 -> 832
- 降低scale
- 调整mosaic
- 增加copy_paste
- AdamW优化器


============================================================
"""


from ultralytics import YOLO
from pathlib import Path
import logging



# ==========================================================
# 项目根目录
# ==========================================================


PROJECT_ROOT = (

    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent

)



# ==========================================================
# 数据集路径
# ==========================================================


DATA_YAML = (

    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "dataset.yaml"

)



# ==========================================================
# 输出名称
# ==========================================================


PROJECT_NAME = "week3_small_object_aug"



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


class SmallObjectTrainer:


    def __init__(self):


        self.model = YOLO(
            "../yolov8n.pt"
        )



    def train(self):


        logging.info(
            "Start Experiment3 Small Object Augmentation"
        )


        results = self.model.train(


            # ----------------------
            # Dataset
            # ----------------------

            data=str(DATA_YAML),


            # ----------------------
            # Training
            # ----------------------

            epochs=100,


            imgsz=832,


            batch=8,


            device=0,


            workers=8,



            # ----------------------
            # Optimizer
            # ----------------------

            optimizer="AdamW",


            lr0=0.001,


            lrf=0.01,


            momentum=0.937,


            weight_decay=0.0005,



            # ----------------------
            # Small Object Augmentation
            # ----------------------


            mosaic=0.5,


            close_mosaic=10,


            copy_paste=0.3,


            mixup=0.1,


            scale=0.2,


            translate=0.1,


            fliplr=0.5,



            # ----------------------
            # Color augmentation
            # ----------------------

            hsv_h=0.015,

            hsv_s=0.7,

            hsv_v=0.4,



            # ----------------------
            # Save
            # ----------------------

            project="runs/train",


            name=PROJECT_NAME,


            exist_ok=True,


            save=True,


            plots=True,



            # ----------------------
            # Reproducibility
            # ----------------------

            seed=42,


        )


        return results




# ==========================================================
# main
# ==========================================================


def main():


    print("="*60)

    print(
        "Week3 Task2"
    )

    print(
        "Experiment 3"
    )

    print(
        "YOLOv8n Small Object Enhancement"
    )

    print("="*60)



    trainer = SmallObjectTrainer()


    trainer.train()



    print()

    print(
        "Experiment3 finished!"
    )

    print(

        "Results saved to:"
        "runs/train/week3_small_object_aug"

    )




if __name__ == "__main__":

    main()