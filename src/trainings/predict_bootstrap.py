"""
predict_bootstrap.py

Purpose:
    Use the Bootstrap YOLO model to automatically predict
    unlabeled images and generate YOLO annotation files.

Author:
    石健明

Date:
    2026-07
"""

import shutil
from pathlib import Path

from ultralytics import YOLO


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(r"E:\Pole_AI_Project")

MODEL_PATH = PROJECT_ROOT / "runs" / "bootstrap" / "weights" / "best.pt"

RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "pole_foreign_objects"

LABELED_IMAGES = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "train"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "bootstrap_predictions"

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"


# ==========================================================
# Create Output Folder
# ==========================================================

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load Model
# ==========================================================

print("Loading model...")
model = YOLO(str(MODEL_PATH))


# ==========================================================
# Prediction
# ==========================================================

total_images = 0
already_labeled = 0
predicted = 0

for class_folder in RAW_DATASET.iterdir():

    if not class_folder.is_dir():
        continue

    print(f"\nProcessing: {class_folder.name}")

    # 保持原来的类别文件夹
    (OUTPUT_IMAGES / class_folder.name).mkdir(parents=True, exist_ok=True)
    (OUTPUT_LABELS / class_folder.name).mkdir(parents=True, exist_ok=True)

    for image_path in class_folder.glob("*.png"):

        total_images += 1

        # 已人工标注
        if (LABELED_IMAGES / image_path.name).exists():
            already_labeled += 1
            continue

        # 保存图片
        dst_image = OUTPUT_IMAGES / class_folder.name / image_path.name
        shutil.copy2(image_path, dst_image)

        # YOLO预测
        results = model.predict(
            source=str(image_path),
            conf=0.25,
            save=False,
            save_txt=False,
            verbose=False
        )

        label_path = (
            OUTPUT_LABELS
            / class_folder.name
            / f"{image_path.stem}.txt"
        )

        with open(label_path, "w", encoding="utf-8") as f:

            boxes = results[0].boxes

            if boxes is not None:

                for box in boxes:

                    cls = int(box.cls.item())

                    x, y, w, h = box.xywhn[0].tolist()

                    f.write(
                        f"{cls} "
                        f"{x:.6f} "
                        f"{y:.6f} "
                        f"{w:.6f} "
                        f"{h:.6f}\n"
                    )

        predicted += 1


# ==========================================================
# Summary
# ==========================================================

print("\n========================================")
print("Bootstrap Prediction Finished")
print("========================================")
print(f"Total images      : {total_images}")
print(f"Already labeled   : {already_labeled}")
print(f"Predicted         : {predicted}")
print(f"Output images     : {OUTPUT_IMAGES}")
print(f"Output labels     : {OUTPUT_LABELS}")
print("========================================")