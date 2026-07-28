"""
Week3 Task4
Final Model Inference

使用最终模型:
YOLOv8n + Class Balanced Augmentation

生成最终推理效果图
"""


from ultralytics import YOLO
from pathlib import Path



# ==========================
# 项目路径
# ==========================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


# 最优模型

MODEL_PATH = (
    PROJECT_ROOT
    /
    "runs"
    /
    "train"
    /
    "week3_aug_v2"
    /
    "weights"
    /
    "best.pt"
)


# 测试图片

SOURCE_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "difficulty_samples"
    /
    "images"
)



# 输出目录

OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "results"
    /
    "week3"
    /
    "final_inference"
)



def main():


    print("="*60)

    print("Week3 Task4")

    print("Final Model Inference")

    print("="*60)



    model = YOLO(
        MODEL_PATH
    )



    results = model.predict(


        source=str(
            SOURCE_DIR
        ),


        imgsz=640,


        conf=0.25,


        iou=0.5,


        save=True,


        save_txt=True,


        save_conf=True,


        project=str(
            OUTPUT_DIR.parent
        ),


        name="final_inference",


        exist_ok=True


    )


    print()

    print("Inference Finished!")

    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )



if __name__ == "__main__":

    main()