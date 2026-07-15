"""
==========================================================
YOLOv8 Baseline Training Script

Week2 Task3

Author:
Project:
==========================================================
"""

from dataclasses import dataclass
from pathlib import Path
import logging

from ultralytics import YOLO


# ==========================================================
# Training Config
# ==========================================================

@dataclass
class TrainConfig:
    """
    Baseline Training Configuration
    """

    # ------------------------------
    # Model
    # ------------------------------
    model_name: str = "yolov8n.pt"

    # ------------------------------
    # Hyper Parameters
    # ------------------------------
    epochs: int = 100

    batch_size: int = 8

    image_size: int = 640

    learning_rate: float = 0.01

    patience: int = 30

    workers: int = 4

    device: int = 0

    seed: int = 42

    pretrained: bool = True

    # ------------------------------
    # Experiment
    # ------------------------------
    experiment_name: str = "baseline_yolov8n"

    project_name: str = "runs/train"


# ==========================================================
# Project Paths
# ==========================================================

@dataclass
class ProjectPaths:

    config: TrainConfig

    def __post_init__(self):

        self.project_root = (
            Path(__file__).resolve().parent.parent.parent
        )

        self.dataset_yaml = (
            self.project_root
            / "data"
            / "processed"
            / "dataset.yaml"
        )

        self.run_dir = (
            self.project_root
            / self.config.project_name
        )


# ==========================================================
# Trainer
# ==========================================================

class BaselineTrainer:

    def __init__(

        self,

        config: TrainConfig,

        paths: ProjectPaths

    ):

        self.config = config

        self.paths = paths

        self.model = YOLO(

            self.config.model_name

        )

    def train(self):

        logging.info("=" * 60)

        logging.info("Start Baseline Training")

        logging.info("=" * 60)

        self.model.train(

            data=str(self.paths.dataset_yaml),

            epochs=self.config.epochs,

            batch=self.config.batch_size,

            imgsz=self.config.image_size,

            lr0=self.config.learning_rate,

            patience=self.config.patience,

            workers=self.config.workers,

            device=self.config.device,

            pretrained=self.config.pretrained,

            seed=self.config.seed,

            project=str(self.paths.run_dir),

            name=self.config.experiment_name,

            exist_ok=True,

            verbose=True

        )

        logging.info("=" * 60)

        logging.info("Training Finished")

        logging.info("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    config = TrainConfig()

    paths = ProjectPaths(config)

    trainer = BaselineTrainer(

        config,

        paths

    )

    trainer.train()


if __name__ == "__main__":

    main()