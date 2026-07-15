"""
==========================================================
YOLOv8 Result Visualization

Week2 Task4

功能：
1. 随机抽取验证集图片
2. 加载 best.pt
3. 绘制 Ground Truth 与 Prediction
4. 保存对比图到 results/week2
==========================================================
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


# ==========================================================
# Visualization Config
# ==========================================================

@dataclass
class VisualizeConfig:
    """
    可视化配置
    """

    # 随机抽取图片数量
    sample_number: int = 10

    # 随机种子（保证实验可复现）
    random_seed: int = 42

    # 预测置信度
    confidence: float = 0.25

    # IoU Threshold
    iou: float = 0.45

    # 图片后缀
    image_suffix = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )


# ==========================================================
# Project Paths
# ==========================================================

@dataclass
class ProjectPaths:
    """
    项目路径管理
    """

    config: VisualizeConfig

    def __post_init__(self):

        # --------------------------------------------------
        # Project Root
        # --------------------------------------------------

        self.project_root = (
            Path(__file__).resolve().parent.parent.parent
        )

        # --------------------------------------------------
        # Validation Dataset
        # --------------------------------------------------

        self.val_images = (
            self.project_root
            / "data"
            / "processed"
            / "split"
            / "images"
            / "val"
        )

        self.val_labels = (
            self.project_root
            / "data"
            / "processed"
            / "split"
            / "labels"
            / "val"
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.best_model = (
            self.project_root
            / "runs"
            / "train"
            / "baseline_yolov8n"
            / "weights"
            / "best.pt"
        )

        # --------------------------------------------------
        # Output Directory
        # --------------------------------------------------

        self.result_dir = (
            self.project_root
            / "results"
            / "week2"
        )

        self.result_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def verify(self):
        """
        检查关键文件是否存在
        """

        if not self.val_images.exists():
            raise FileNotFoundError(
                f"Validation images not found:\n{self.val_images}"
            )

        if not self.val_labels.exists():
            raise FileNotFoundError(
                f"Validation labels not found:\n{self.val_labels}"
            )

        if not self.best_model.exists():
            raise FileNotFoundError(
                f"best.pt not found:\n{self.best_model}"
            )

# ==========================================================
# Result Visualizer
# ==========================================================

class ResultVisualizer:
    """
    YOLOv8结果可视化

    功能：
    1. 随机抽取验证集图片
    2. 绘制Ground Truth
    3. 绘制Prediction
    4. 保存结果
    """

    CLASS_NAMES = [
        "bird_nest",
        "balloon",
        "plastic_bag",
        "other_foreign_object"
    ]

    def __init__(
        self,
        config: VisualizeConfig,
        paths: ProjectPaths
    ):

        self.config = config
        self.paths = paths

        self.model = YOLO(
            str(paths.best_model)
        )

    # ------------------------------------------------------

    @staticmethod
    def yolo_to_xyxy(
        x,
        y,
        w,
        h,
        img_w,
        img_h
    ):

        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)

        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)

        return x1, y1, x2, y2

    # ------------------------------------------------------

    def draw_ground_truth(
        self,
        image,
        label_path: Path
    ):

        if not label_path.exists():
            return

        h, w = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:

            for line in f:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                cls = int(values[0])

                xc = float(values[1])
                yc = float(values[2])
                bw = float(values[3])
                bh = float(values[4])

                x1, y1, x2, y2 = self.yolo_to_xyxy(
                    xc,
                    yc,
                    bw,
                    bh,
                    w,
                    h
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"GT:{self.CLASS_NAMES[cls]}",
                    (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

    # ------------------------------------------------------

    def draw_prediction(
        self,
        image
    ):

        results = self.model.predict(

            source=image,

            conf=self.config.confidence,

            iou=self.config.iou,

            verbose=False

        )

        result = results[0]

        if result.boxes is None:
            return

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls.item())

            conf = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 0, 255),

                2

            )

            cv2.putText(

                image,

                f"Pred:{self.CLASS_NAMES[cls]} {conf:.2f}",

                (x1, y2 + 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 0, 255),

                2

            )

    # ------------------------------------------------------

    def visualize(self):

        random.seed(
            self.config.random_seed
        )

        image_list = [

            p

            for p in self.paths.val_images.iterdir()

            if p.suffix.lower()
            in self.config.image_suffix

        ]

        if len(image_list) == 0:

            raise RuntimeError(
                "Validation dataset is empty."
            )

        sample_num = min(

            self.config.sample_number,

            len(image_list)

        )

        selected = random.sample(

            image_list,

            sample_num

        )

        logging.info(
            "Selected %d validation images.",
            sample_num
        )

        for image_path in selected:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            label_path = (
                self.paths.val_labels
                / f"{image_path.stem}.txt"
            )

            self.draw_ground_truth(
                image,
                label_path
            )

            self.draw_prediction(
                image
            )

            save_path = (

                self.paths.result_dir

                / f"comparison_{image_path.stem}.jpg"

            )

            cv2.imwrite(

                str(save_path),

                image

            )

            logging.info(

                "Saved: %s",

                save_path.name

            )

# ==========================================================
# Main
# ==========================================================

def main():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    logging.info("=" * 60)
    logging.info("Start Visualization")
    logging.info("=" * 60)

    # ------------------------------------------------------
    # Initialize
    # ------------------------------------------------------

    config = VisualizeConfig()

    paths = ProjectPaths(config)

    paths.verify()

    visualizer = ResultVisualizer(

        config,

        paths

    )

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------

    visualizer.visualize()

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    image_number = len(

        [

            p

            for p in paths.val_images.iterdir()

            if p.suffix.lower()

            in config.image_suffix

        ]

    )

    saved_number = len(

        list(

            paths.result_dir.glob(

                "comparison_*.jpg"

            )

        )

    )

    logging.info("=" * 60)

    logging.info("Visualization Finished")

    logging.info("=" * 60)

    logging.info(

        "Validation Images : %d",

        image_number

    )

    logging.info(

        "Random Selected   : %d",

        config.sample_number

    )

    logging.info(

        "Saved Results     : %d",

        saved_number

    )

    logging.info(

        "Output Directory  : %s",

        paths.result_dir

    )

    logging.info("=" * 60)


if __name__ == "__main__":

    main()