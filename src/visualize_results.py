"""
File:
    visualize_results.py

Description:
    Visualize YOLO detection results.
    Compare ground truth labels and model predictions.

Features:
    1. Randomly select validation images
    2. Draw ground truth bounding boxes
    3. Draw model prediction bounding boxes
    4. Save comparison images


"""


from pathlib import Path
import random
import cv2

from ultralytics import YOLO


# ======================================================
# Project Path
# ======================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# validation images

IMAGE_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "final_dataset"
    /
    "images"
    /
    "val"
)


# validation labels

LABEL_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "final_dataset"
    /
    "labels"
    /
    "val"
)


# baseline model

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


# output

OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "results"
    /
    "week3"
    /
    "best_model_comparison"
)


# classes

CLASS_NAMES = [
    "bird_nest",
    "balloon",
    "plastic_bag",
    "other_foreign_object"
]


# ======================================================
# YOLO label conversion
# ======================================================

def yolo_to_xyxy(
        label,
        img_width,
        img_height
):

    """
    Convert YOLO format:

    class x_center y_center width height

    to:

    x1 y1 x2 y2

    """


    cls, xc, yc, w, h = label


    x1 = int(
        (xc - w / 2)
        *
        img_width
    )

    y1 = int(
        (yc - h / 2)
        *
        img_height
    )

    x2 = int(
        (xc + w / 2)
        *
        img_width
    )

    y2 = int(
        (yc + h / 2)
        *
        img_height
    )


    return (
        int(cls),
        x1,
        y1,
        x2,
        y2
    )


# ======================================================
# Draw ground truth
# ======================================================

def draw_ground_truth(
        image,
        label_path
):


    if not label_path.exists():
        return image


    h, w = image.shape[:2]


    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        labels = f.readlines()


    for line in labels:


        data = list(
            map(
                float,
                line.strip().split()
            )
        )


        cls,x1,y1,x2,y2 = yolo_to_xyxy(
            data,
            w,
            h
        )


        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            image,
            "GT:"
            +
            CLASS_NAMES[cls],
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )


    return image



# ======================================================
# Draw prediction
# ======================================================

def draw_prediction(
        image,
        result
):


    boxes = result.boxes


    for box in boxes:


        cls = int(
            box.cls[0]
        )


        conf = float(
            box.conf[0]
        )


        x1,y1,x2,y2 = (
            map(
                int,
                box.xyxy[0]
            )
        )


        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (0,0,255),
            2
        )


        cv2.putText(
            image,
            f"P:{CLASS_NAMES[cls]} {conf:.2f}",
            (x1,y2+20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )


    return image



# ======================================================
# Main
# ======================================================

def main():


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("="*60)
    print("Baseline Visualization")
    print("="*60)


    # load model

    model = YOLO(
        MODEL_PATH
    )


    images = list(
        IMAGE_DIR.glob(
            "*.png"
        )
    )


    if len(images) < 10:

        selected_images = images

    else:

        selected_images = random.sample(
            images,
            10
        )


    print(
        f"Selected images: {len(selected_images)}"
    )


    for idx, img_path in enumerate(
        selected_images,
        start=1
    ):


        image = cv2.imread(
            str(img_path)
        )


        # prediction

        result = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False
        )[0]


        # draw GT

        label_path = (
            LABEL_DIR
            /
            (
                img_path.stem
                +
                ".txt"
            )
        )


        output = image.copy()


        output = draw_ground_truth(
            output,
            label_path
        )


        output = draw_prediction(
            output,
            result
        )


        save_path = (
            OUTPUT_DIR
            /
            f"comparison_{idx:03d}.jpg"
        )


        cv2.imwrite(
            str(save_path),
            output
        )


        print(
            f"Saved: {save_path}"
        )


    print("="*60)
    print("Visualization Finished")
    print("="*60)



if __name__ == "__main__":

    main()