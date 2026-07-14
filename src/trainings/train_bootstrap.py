"""
train_bootstrap.py

Purpose:
    Train an initial YOLOv8 model using the manually labeled images.
    The trained model will be used for model-assisted annotation of the
    remaining unlabeled images.

Author:
    Mingh

Date:
    2026-07
"""

from pathlib import Path

from ultralytics import YOLO


def main():
    # ==========================================================
    # Project Path
    # ==========================================================
    project_root = Path(r"E:\Pole_AI_Project")

    dataset_yaml = (
        project_root
        / "configs"
        / "bootstrap_dataset.yaml"
    )

    # ==========================================================
    # Load Pretrained Model
    # ==========================================================
    model = YOLO("yolov8n.pt")

    # ==========================================================
    # Start Training
    # ==========================================================
    model.train(
        data=str(dataset_yaml),

        # ---------- Training ----------
        epochs=100,
        batch=8,
        imgsz=640,

        # ---------- Hardware ----------
        device=0,
        workers=4,

        # ---------- Optimization ----------
        lr0=0.01,
        optimizer="auto",

        # ---------- Early Stop ----------
        patience=20,

        # ---------- Project ----------
        project=str(project_root / "runs"),
        name="bootstrap",

        # ---------- Save ----------
        save=True,
        save_period=-1,

        # ---------- Visualization ----------
        plots=True,
        verbose=True
    )

    print("\nBootstrap training completed.")


if __name__ == "__main__":
    main()