"""
============================================================
Week3 Task4

Baseline vs Optimized Model Comparison Prediction


功能:

1. 加载Baseline YOLOv8n模型
2. 加载优化YOLOv8n模型
3. 使用完全相同test图片进行推理
4. 分别保存检测结果
5. 用于困难样本可视化分析


输出:

results/week3/comparison/

    baseline/
        xxx.jpg

    optimized/
        xxx.jpg


============================================================
"""


from pathlib import Path
from ultralytics import YOLO
import logging
import shutil



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



# ===============================
# 模型路径
# ===============================


BASELINE_MODEL = (

    PROJECT_ROOT
    /
    "runs"
    /
    "train"
    /
    "baseline_yolov8n"
    /
    "weights"
    /
    "best.pt"

)



OPTIMIZED_MODEL = (

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



# ===============================
# 测试集路径
# ===============================


TEST_DIR = (

    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "split"
    /
    "images"
    /
    "test"

)



# ===============================
# 输出路径
# ===============================


OUTPUT_DIR = (

    PROJECT_ROOT
    /
    "results"
    /
    "week3"
    /
    "comparison"

)



BASELINE_OUTPUT = OUTPUT_DIR / "baseline"


OPTIMIZED_OUTPUT = OUTPUT_DIR / "optimized"




# ==========================================================
# 日志
# ==========================================================


logging.basicConfig(

    level=logging.INFO,

    format=
    "%(asctime)s - %(levelname)s - %(message)s"

)




# ==========================================================
# 推理类
# ==========================================================


class ModelComparator:


    def __init__(self):


        self.baseline = None

        self.optimized = None



    def load_models(self):


        logging.info(
            "Loading models..."
        )


        self.baseline = YOLO(
            str(BASELINE_MODEL)
        )


        self.optimized = YOLO(
            str(OPTIMIZED_MODEL)
        )


        logging.info(
            "Models loaded successfully"
        )



    def prepare_output(self):


        """
        清空旧结果
        """

        if OUTPUT_DIR.exists():

            shutil.rmtree(
                OUTPUT_DIR
            )


        BASELINE_OUTPUT.mkdir(
            parents=True,
            exist_ok=True
        )


        OPTIMIZED_OUTPUT.mkdir(
            parents=True,
            exist_ok=True
        )



    def predict_single_model(
            self,
            model,
            output_dir,
            image_path
    ):


        """
        单张图片推理
        """


        results = model.predict(

            source=str(image_path),

            imgsz=640,

            conf=0.25,

            iou=0.5,

            save=True,

            save_txt=True,

            save_conf=True,

            project=str(output_dir),

            name="",

            exist_ok=True,

            device=0

        )


        return results



    def run(self):


        images = list(

            TEST_DIR.glob(
                "*.png"
            )

        )


        if len(images) == 0:

            raise FileNotFoundError(

                f"No test images found: {TEST_DIR}"

            )


        logging.info(

            f"Test images number: {len(images)}"

        )


        for index, image in enumerate(images):


            logging.info(

                f"[{index+1}/{len(images)}] "
                f"Processing {image.name}"

            )



            # ----------------------
            # baseline
            # ----------------------


            self.predict_single_model(

                self.baseline,

                BASELINE_OUTPUT,

                image

            )



            # ----------------------
            # optimized
            # ----------------------


            self.predict_single_model(

                self.optimized,

                OPTIMIZED_OUTPUT,

                image

            )



        logging.info(
            "Comparison prediction finished!"
        )




# ==========================================================
# main
# ==========================================================


def main():


    print("="*60)

    print(
        "Week3 Task4"
    )

    print(
        "Baseline vs Optimized Comparison"
    )

    print("="*60)



    comparator = ModelComparator()



    comparator.load_models()


    comparator.prepare_output()


    comparator.run()



    print()

    print(
        "Results saved:"
    )

    print(
        OUTPUT_DIR
    )




if __name__ == "__main__":

    main()