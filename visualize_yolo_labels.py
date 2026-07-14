"""
visualize_yolo_labels.py

Purpose:
    Visualize YOLO annotations on images.

Author:
    石健明

Date:
    2026-07
"""

from pathlib import Path
import cv2


# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(r"E:\Pole_AI_Project")

DATASET_DIR = PROJECT_ROOT / "data" / "bootstrap_predictions"

CLASS_NAMES = [
    "bird_nest",
    "balloon",
    "plastic_bag",
    "other_foreign_object"
]


# =====================================================
# Draw YOLO Boxes
# =====================================================

def draw_boxes(image_path: Path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot read image: {image_path.name}")
        return

    h, w = image.shape[:2]

    label_path = image_path.with_suffix(".txt")

    if not label_path.exists():
        print(f"No label: {image_path.name}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        cls, xc, yc, bw, bh = map(float, line.split())

        cls = int(cls)

        # YOLO -> Pixel
        xc *= w
        yc *= h
        bw *= w
        bh *= h

        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = CLASS_NAMES[cls]

        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO Visualization", image)

    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

    return key


# =====================================================
# Main
# =====================================================

def main():

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    }

    images = sorted([
        p for p in DATASET_DIR.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images.\n")

    print("Controls:")
    print("  SPACE : Next image")
    print("  ESC   : Exit\n")

    for image_path in images:

        print(f"Viewing: {image_path.name}")

        key = draw_boxes(image_path)

        if key == 27:
            break


if __name__ == "__main__":
    main()