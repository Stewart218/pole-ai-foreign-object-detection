"""
==========================================================
YOLOv8 Baseline Model Evaluation

Week2 Task5

功能：
1. 调用YOLOv8官方val接口
2. 在验证集评估模型
3. 获取Precision / Recall / mAP指标
4. 生成evaluation_report_v1.txt

==========================================================
"""


from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime


from ultralytics import YOLO



# ==========================================================
# Evaluation Config
# ==========================================================

@dataclass
class EvaluateConfig:
    """
    模型评估配置
    """

    # 验证批次大小
    batch_size: int = 16

    # 输入尺寸
    imgsz: int = 640

    # 数据加载线程
    workers: int = 4

    # 置信度阈值
    conf: float = 0.001

    # IoU阈值
    iou: float = 0.6



# ==========================================================
# Project Paths
# ==========================================================

@dataclass
class ProjectPaths:
    """
    项目路径管理
    """

    config: EvaluateConfig


    def __post_init__(self):

        # 项目根目录

        self.project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            .parent
        )


        # ------------------------------
        # Dataset
        # ------------------------------

        self.dataset_yaml = (

            self.project_root

            / "data"

            / "processed"

            / "dataset.yaml"

        )


        # ------------------------------
        # Model
        # ------------------------------

        self.best_model = (

            self.project_root

            / "runs"

            / "train"

            / "baseline_yolov8n"

            / "weights"

            / "best.pt"

        )


        # ------------------------------
        # Output
        # ------------------------------

        self.result_dir = (

            self.project_root

            / "results"

            / "week2"

        )


        self.report_file = (

            self.result_dir

            / "evaluation_report_v1.txt"

        )


        self.result_dir.mkdir(

            parents=True,

            exist_ok=True

        )



    def verify(self):
        """
        检查路径
        """


        if not self.dataset_yaml.exists():

            raise FileNotFoundError(

                f"dataset.yaml not found:\n"
                f"{self.dataset_yaml}"

            )


        if not self.best_model.exists():

            raise FileNotFoundError(

                f"best.pt not found:\n"
                f"{self.best_model}"

            )



# ==========================================================
# Model Evaluator
# ==========================================================

class ModelEvaluator:
    """
    YOLOv8模型评估器
    """


    def __init__(

        self,

        config: EvaluateConfig,

        paths: ProjectPaths

    ):


        self.config = config

        self.paths = paths


        self.model = YOLO(

            str(

                self.paths.best_model

            )

        )

    # ==========================================================
    # Evaluation
    # ==========================================================

    def evaluate(self):
        """
        调用YOLOv8官方验证接口
        """

        logging.info(
            "Start model evaluation..."
        )

        metrics = self.model.val(

            data=str(
                self.paths.dataset_yaml
            ),

            imgsz=self.config.imgsz,

            batch=self.config.batch_size,

            workers=self.config.workers,

            conf=self.config.conf,

            iou=self.config.iou,

            verbose=True

        )

        # ------------------------------
        # Extract Metrics
        # ------------------------------

        evaluation_result = {

            "Precision": float(

                metrics.box.mp

            ),

            "Recall": float(

                metrics.box.mr

            ),

            "mAP50": float(

                metrics.box.map50

            ),

            "mAP50-95": float(

                metrics.box.map

            )

        }

        return evaluation_result

    # ==========================================================
    # Report Generator
    # ==========================================================

    def generate_report(

            self,

            results: dict

    ):
        """
        生成评估报告
        """

        with open(

                self.paths.report_file,

                "w",

                encoding="utf-8"

        ) as f:
            f.write(

                "=" * 60

                + "\n"

            )

            f.write(

                "YOLOv8 Baseline Model Evaluation Report\n"

            )

            f.write(

                "=" * 60

                + "\n\n"

            )

            f.write(

                "Model\n"

            )

            f.write(

                "-" * 40

                + "\n"

            )

            f.write(

                f"{self.paths.best_model.name}\n\n"

            )

            f.write(

                "Dataset\n"

            )

            f.write(

                "-" * 40

                + "\n"

            )

            f.write(

                "Validation Set\n\n"

            )

            f.write(

                "Evaluation Time\n"

            )

            f.write(

                "-" * 40

                + "\n"

            )

            f.write(

                datetime.now().strftime(

                    "%Y-%m-%d %H:%M:%S"

                )

                + "\n\n"

            )

            f.write(

                "Metrics\n"

            )

            f.write(

                "-" * 40

                + "\n"

            )

            f.write(

                f"Precision     : "
                f"{results['Precision']:.4f}\n"

            )

            f.write(

                f"Recall        : "
                f"{results['Recall']:.4f}\n"

            )

            f.write(

                f"mAP@0.5       : "
                f"{results['mAP50']:.4f}\n"

            )

            f.write(

                f"mAP@0.5:0.95  : "
                f"{results['mAP50-95']:.4f}\n\n"

            )

            f.write(

                "Performance Analysis\n"

            )

            f.write(

                "-" * 40

                + "\n"

            )

            f.write(

                "Advantages:\n"

            )

            f.write(

                "- Baseline training pipeline completed successfully.\n"

            )

            f.write(

                "- Model can detect part of foreign object targets.\n\n"

            )

            f.write(

                "Limitations:\n"

            )

            f.write(

                "- Dataset scale is relatively small.\n"

            )

            f.write(

                "- Recall indicates missing detection problems remain.\n"

            )

            f.write(

                "- Small targets and complex backgrounds affect detection performance.\n"

            )

            f.write(

                "- mAP50-95 shows localization accuracy still needs improvement.\n\n"

            )

            f.write(

                "Conclusion:\n"

            )

            f.write(

                "The baseline model is used as the reference model "

                "for further optimization in later stages.\n"

            )

            f.write(

                "\n"

                + "=" * 60

            )

        logging.info(

            "Evaluation report saved:\n%s",

            self.paths.report_file

        )

# ==========================================================
    # Main
# ==========================================================

def main():


    logging.basicConfig(

        level=logging.INFO,

        format=(

            "%(asctime)s - "

            "%(levelname)s - "

            "%(message)s"

        )

    )


    logging.info(

        "=" * 60

    )

    logging.info(

        "YOLOv8 Evaluation Start"

    )

    logging.info(

        "=" * 60

    )


    config = EvaluateConfig()


    paths = ProjectPaths(

        config

    )


    paths.verify()



    evaluator = ModelEvaluator(

        config,

        paths

    )


    results = evaluator.evaluate()


    evaluator.generate_report(

        results

    )


    logging.info(

        "=" * 60

    )


    logging.info(

        "Evaluation Completed"

    )


    logging.info(

        "Precision : %.4f",

        results["Precision"]

    )


    logging.info(

        "Recall : %.4f",

        results["Recall"]

    )


    logging.info(

        "mAP50 : %.4f",

        results["mAP50"]

    )


    logging.info(

        "mAP50-95 : %.4f",

        results["mAP50-95"]

    )


    logging.info(

        "=" * 60

    )



if __name__ == "__main__":

    main()