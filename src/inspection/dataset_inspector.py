"""
===========================================================
Task 2 - Dataset Inspector
Author : ChatGPT
Project: Pole AI Foreign Object Detection

Functions
---------
1. Dataset size statistics
2. Class distribution statistics
3. Image format statistics
4. Image resolution statistics
5. Generate Markdown report

This version is for RAW dataset only.
(No YOLO labels required.)

Compatible with Windows Chinese paths.
===========================================================
"""

from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import pandas as pd


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pole_foreign_objects"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# Image Reader
# Support Chinese Path
# =========================================================

def read_image(image_path: Path):
    """
    Read image using cv2.imdecode.
    This method supports Chinese file paths on Windows.
    """

    try:

        image_bytes = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )

        if image_bytes.size == 0:
            return None

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None


# =========================================================
# Dataset Inspector
# =========================================================

class DatasetInspector:

    def __init__(self):

        self.total_images = 0

        self.class_counter = Counter()

        self.format_counter = Counter()

        self.resolution_counter = Counter()

        self.records = []

    # -----------------------------------------------------

    def scan_dataset(self):

        if not DATASET_DIR.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n{DATASET_DIR}"
            )

        print("=" * 60)
        print("Scanning Dataset...")
        print("=" * 60)

        for class_folder in sorted(DATASET_DIR.iterdir()):

            if not class_folder.is_dir():
                continue

            class_name = class_folder.name

            print(f"Class: {class_name}")

            self.scan_class(
                class_folder,
                class_name
            )

        print()

        print("Dataset Scan Finished.")

    # -----------------------------------------------------

    def scan_class(
            self,
            class_folder,
            class_name
    ):

        for image_path in sorted(class_folder.iterdir()):

            if not image_path.is_file():
                continue

            suffix = image_path.suffix.lower()

            if suffix not in IMAGE_EXTENSIONS:
                continue

            image = read_image(image_path)

            if image is None:
                continue

            height, width = image.shape[:2]

            self.total_images += 1

            self.class_counter[class_name] += 1

            self.format_counter[suffix] += 1

            resolution = f"{width}×{height}"

            self.resolution_counter[
                resolution
            ] += 1

            self.records.append(
                {
                    "Class": class_name,
                    "Filename": image_path.name,
                    "Format": suffix,
                    "Width": width,
                    "Height": height,
                    "Resolution": resolution
                }
            )

    # -----------------------------------------------------

    def print_statistics(self):

        print()

        print("=" * 60)
        print("Dataset Statistics")
        print("=" * 60)

        print(f"Total Images : {self.total_images}")

        print()

        print("Class Distribution")

        for cls, num in self.class_counter.items():

            percent = (
                num /
                self.total_images *
                100
            )

            print(
                f"{cls:<10}"
                f"{num:>6}"
                f" ({percent:.2f}%)"
            )

        print()

        print("Image Formats")

        for fmt, num in self.format_counter.items():

            print(
                f"{fmt:<8}"
                f"{num}"
            )

        print()

        print("Top 10 Resolutions")

        for resolution, number in self.resolution_counter.most_common(10):

            print(
                f"{resolution:<15}"
                f"{number}"
            )

    # -----------------------------------------------------

    def export_csv(self):

        dataframe = pd.DataFrame(
            self.records
        )

        dataframe.to_csv(
            REPORT_DIR / "image_information.csv",
            index=False,
            encoding="utf-8-sig"
        )

        class_df = pd.DataFrame(
            {
                "Class":
                    list(self.class_counter.keys()),
                "Images":
                    list(self.class_counter.values())
            }
        )

        class_df.to_csv(
            REPORT_DIR / "dataset_statistics.csv",
            index=False,
            encoding="utf-8-sig"
        )
    # -----------------------------------------------------

    def generate_markdown_report(self):

        report_path = REPORT_DIR / "dataset_report.md"

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("# Dataset Inspection Report\n\n")

            f.write("## 1. Dataset Overview\n\n")

            f.write(
                f"- Total Images: **{self.total_images}**\n\n"
            )

            f.write("## 2. Class Distribution\n\n")

            f.write("| Class | Images | Percentage |\n")
            f.write("|------|-------:|-----------:|\n")

            for cls, num in self.class_counter.items():

                percent = (
                    num /
                    self.total_images *
                    100
                )

                f.write(
                    f"| {cls} | {num} | {percent:.2f}% |\n"
                )

            f.write("\n")

            f.write("## 3. Image Format Statistics\n\n")

            f.write("| Format | Count |\n")
            f.write("|-------|------:|\n")

            for fmt, num in self.format_counter.items():

                f.write(
                    f"| {fmt} | {num} |\n"
                )

            f.write("\n")

            f.write("## 4. Resolution Statistics\n\n")

            dataframe = pd.DataFrame(
                self.records
            )

            if not dataframe.empty:

                stats = dataframe[["Width", "Height"]].describe()

                f.write("| Statistic | Width | Height |\n")
                f.write("|-----------|------:|-------:|\n")

                for index, row in stats.iterrows():
                    f.write(
                        f"| {index} | "
                        f"{row['Width']:.2f} | "
                        f"{row['Height']:.2f} |\n"
                    )

                f.write("\n")

            f.write("## 5. Top 10 Resolutions\n\n")

            f.write("| Resolution | Count |\n")
            f.write("|-----------|------:|\n")

            for resolution, number in (
                    self.resolution_counter.most_common(10)
            ):

                f.write(
                    f"| {resolution} | {number} |\n"
                )

        print()
        print("=" * 60)
        print("Markdown report generated.")
        print(report_path)

    # -----------------------------------------------------

    def run(self) -> List[ImageInfo]:

        images = self.scan_dataset()

        self.print_statistics()

        self.export_csv()

        self.generate_markdown_report()

        return images


# =========================================================
# Main
# =========================================================

def main():

    print()

    print("=" * 60)
    print("Pole AI Dataset Inspector")
    print("=" * 60)

    inspector = DatasetInspector()

    inspector.run()

    print()

    print("=" * 60)
    print("Inspection Finished Successfully.")
    print("=" * 60)

    print()

    print("Output Directory:")

    print(REPORT_DIR)

    print()

    print("Generated Files:")

    print("  image_information.csv")

    print("  dataset_statistics.csv")

    print("  dataset_report.md")

    print()

    print("Task 2 Completed.")


if __name__ == "__main__":

    main()