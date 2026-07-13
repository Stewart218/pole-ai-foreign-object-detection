"""
======================================================================
Enterprise YOLO Dataset Preprocessing Pipeline
----------------------------------------------------------------------
Project : Pole AI Foreign Object Detection
Author  : Stewart218
Version : 2.0.0
Python  : 3.9
======================================================================

Week2 - Task1

Functions
---------
1. Dataset Scanning
2. Image Cleaning
3. Label Validation
4. Image Standardization
5. Data Augmentation
6. Dataset Balancing
7. Report Generation

Directory
---------
data/
└── yolo_dataset/
    ├── images/
    └── labels/

Output
------
data/
└── processed/
"""

from __future__ import annotations

import cv2
import hashlib
import logging
import random
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==========================================================
# Random Seed
# ==========================================================

GLOBAL_RANDOM_SEED = 42

random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)


# ==========================================================
# Logger
# ==========================================================

LOGGER = logging.getLogger("Preprocess")

LOGGER.setLevel(logging.INFO)

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

_console = logging.StreamHandler()
_console.setFormatter(_formatter)

LOGGER.handlers.clear()
LOGGER.addHandler(_console)


# ==========================================================
# Basic Utility
# ==========================================================

def print_title(title: str) -> None:
    """
    Print section title.

    Parameters
    ----------
    title : str
        Title text.
    """

    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info(title)
    LOGGER.info("=" * 70)


def ensure_dir(directory: Path) -> None:
    """
    Create directory recursively.

    Parameters
    ----------
    directory : Path
        Target directory.
    """

    directory.mkdir(parents=True, exist_ok=True)


def calculate_md5(file_path: Path) -> str:
    """
    Calculate file MD5.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    str
    """

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(4096)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()


# ==========================================================
# OpenCV (Chinese Path Support)
# ==========================================================

def cv_imread(image_path: Path):
    """
    Read image with Chinese path support.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    numpy.ndarray
    """

    buffer = np.fromfile(str(image_path), dtype=np.uint8)

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return image


def cv_imwrite(image_path: Path, image) -> None:
    """
    Save image with Chinese path support.

    Parameters
    ----------
    image_path : Path

    image : ndarray
    """

    success, encoded = cv2.imencode(image_path.suffix, image)

    if success:

        encoded.tofile(str(image_path))

# ==========================================================
# Project Paths
# ==========================================================

class ProjectPaths:
    """
    Project directory manager.

    Project Structure
    -----------------
    Pole_AI_Project/
    ├── data/
    │   ├── raw/
    │   ├── yolo_dataset/
    │   │   ├── images/
    │   │   └── labels/
    │   └── processed/
    │       ├── images/
    │       ├── labels/
    │       ├── reports/
    │       ├── visualization/
    │       └── cache/
    ├── src/
    │   └── trainings/
    │       └── preprocess.py
    └── reports/
    """

    def __init__(self) -> None:

        self.project_root = Path(__file__).resolve().parents[2]

        # ---------------- Dataset ---------------- #

        self.dataset_root = (
            self.project_root
            / "data"
            / "yolo_dataset"
        )

        self.image_root = (
            self.dataset_root
            / "images"
        )

        self.label_root = (
            self.dataset_root
            / "labels"
        )

        # ---------------- Output ---------------- #

        self.processed_root = (
            self.project_root
            / "data"
            / "processed"
        )

        self.output_image_root = (
            self.processed_root
            / "images"
        )

        self.output_label_root = (
            self.processed_root
            / "labels"
        )

        self.report_root = (
            self.processed_root
            / "reports"
        )

        self.visualization_root = (
            self.processed_root
            / "visualization"
        )

        self.cache_root = (
            self.processed_root
            / "cache"
        )

    # ------------------------------------------------------

    def initialize(self) -> None:
        """
        Create output folders.
        """

        folders = [

            self.processed_root,

            self.output_image_root,

            self.output_label_root,

            self.report_root,

            self.visualization_root,

            self.cache_root

        ]

        for folder in folders:

            ensure_dir(folder)

        LOGGER.info("Project folders initialized.")


# ==========================================================
# Global Configuration
# ==========================================================

@dataclass
class PreprocessConfig:
    """
    Global configuration.

    Notes
    -----
    This class stores only parameters.

    No project paths should appear here.
    """

    # ---------------- Random ---------------- #

    random_seed: int = 42

    # ---------------- Image ---------------- #

    target_size: Tuple[int, int] = (
        640,
        640,
    )

    normalize: bool = True

    keep_aspect_ratio: bool = True

    interpolation: int = cv2.INTER_LINEAR

    # ---------------- Cleaning ---------------- #

    blur_threshold: float = 100.0

    remove_duplicate: bool = True

    remove_invalid: bool = True

    # ---------------- Augmentation ---------------- #

    enable_augmentation: bool = True

    augment_times: int = 2

    # ---------------- Visualization ---------------- #

    save_visualization: bool = True

    # ---------------- Report ---------------- #

    save_csv: bool = True

    save_markdown: bool = True

    # ---------------- Supported Image ---------------- #

    supported_suffix: Tuple[str, ...] = (

        ".jpg",

        ".jpeg",

        ".png",

        ".bmp",

        ".tif",

        ".tiff",

    )


# ==========================================================
# Data Models
# ==========================================================

@dataclass
class BoundingBox:
    """
    YOLO Bounding Box.
    """

    class_id: int

    x_center: float

    y_center: float

    width: float

    height: float


# ----------------------------------------------------------


@dataclass
class ImageInfo:
    """
    Metadata of one image.

    This object is shared by all modules.
    """

    image_path: Path

    label_path: Optional[Path]

    class_name: str

    width: int = 0

    height: int = 0

    channels: int = 3

    image_format: str = ""

    file_size: int = 0

    aspect_ratio: float = 0.0

    bbox_count: int = 0

    md5: str = ""

    has_label: bool = False

    label_loaded: bool = False

    is_duplicate: bool = False

    is_blurry: bool = False

    is_valid: bool = True

    error_message: str = ""

    label_error: str = ""

    bboxes: List[BoundingBox] = field(
        default_factory=list
    )


# ----------------------------------------------------------


@dataclass
class DatasetStatistics:
    """
    Dataset statistics.

    Generated after scanning.
    """

    total_images: int = 0

    labeled_images: int = 0

    unlabeled_images: int = 0

    missing_labels: int = 0

    orphan_labels: int = 0

    duplicate_images: int = 0

    blurry_images: int = 0

    invalid_images: int = 0

    total_bboxes: int = 0

    class_distribution: Dict[str, int] = field(
        default_factory=dict
    )

    image_formats: Dict[str, int] = field(
        default_factory=dict
    )

    resolutions: Dict[str, int] = field(
        default_factory=dict
    )


# ==========================================================
# Global Objects
# ==========================================================

CONFIG = PreprocessConfig()

PATHS = ProjectPaths()

# ==========================================================
# Dataset Scanner
# ==========================================================

class DatasetScanner:
    """
    Discover dataset structure.

    Responsibilities
    ----------------
    1. Discover class folders
    2. Discover image files
    3. Match label files
    4. Build ImageInfo list

    Notes
    -----
    This class DOES NOT

    - read image
    - calculate md5
    - read bbox
    - collect statistics

    Those jobs belong to MetadataCollector.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: PreprocessConfig,
    ) -> None:

        self.paths = paths
        self.config = config

    # ------------------------------------------------------

    def discover_categories(self) -> List[Path]:
        """
        Discover all category folders.

        Returns
        -------
        List[Path]
            Sorted category folders.
        """

        print_title("Discover Categories")

        categories = [

            folder

            for folder in self.paths.image_root.iterdir()

            if folder.is_dir()

        ]

        categories.sort(
            key=lambda x: x.name
        )

        LOGGER.info(
            f"Found {len(categories)} categories."
        )

        for folder in categories:

            LOGGER.info(
                f"  ├── {folder.name}"
            )

        return categories

    # ------------------------------------------------------

    def discover_images(
        self,
        category_dir: Path,
    ) -> List[Path]:
        """
        Discover image files.

        Parameters
        ----------
        category_dir : Path

        Returns
        -------
        List[Path]
        """

        image_files: List[Path] = []

        for file in category_dir.rglob("*"):

            if not file.is_file():

                continue

            if (
                file.suffix.lower()
                not in self.config.supported_suffix
            ):

                continue

            image_files.append(file)

        image_files.sort()

        LOGGER.info(

            f"{category_dir.name}: "

            f"{len(image_files)} images"

        )

        return image_files

    # ------------------------------------------------------

    def find_label(
        self,
        image_path: Path,
    ) -> Optional[Path]:
        """
        Match YOLO label.

        Parameters
        ----------
        image_path : Path

        Returns
        -------
        Optional[Path]
        """

        label_path = (

            self.paths.label_root

            / f"{image_path.stem}.txt"

        )

        if label_path.exists():

            return label_path

        return None

    # ------------------------------------------------------

    def build_image_info(
        self,
        image_path: Path,
        label_path: Optional[Path],
        class_name: str,
    ) -> ImageInfo:
        """
        Build ImageInfo object.

        Notes
        -----
        Only build file relationships.

        Do NOT read image here.
        """

        return ImageInfo(

            image_path=image_path,

            label_path=label_path,

            class_name=class_name,

            has_label=label_path is not None,

            image_format=image_path.suffix.lower(),

        )

    # ------------------------------------------------------

    def scan(self) -> List[ImageInfo]:
        """
        Scan whole dataset.

        Returns
        -------
        List[ImageInfo]

        Notes
        -----
        Only discover dataset.

        Metadata will be collected
        by MetadataCollector.
        """

        print_title("Scanning Dataset")

        image_infos: List[ImageInfo] = []

        categories = self.discover_categories()

        for category in categories:

            image_files = self.discover_images(category)

            for image_path in tqdm(

                image_files,

                desc=category.name,

                leave=False,

            ):

                label_path = self.find_label(image_path)

                info = self.build_image_info(

                    image_path=image_path,

                    label_path=label_path,

                    class_name=category.name,

                )

                image_infos.append(info)

        LOGGER.info("")

        LOGGER.info(

            f"Dataset discovery completed."

        )

        LOGGER.info(

            f"Discovered images: "

            f"{len(image_infos)}"

        )

        return image_infos

    # ------------------------------------------------------

    def run(
        self,
    ) -> List[ImageInfo]:
        """
        Scan the dataset and return all image information.

        Returns
        -------
        List[ImageInfo]
            All discovered images.
        """

        print_title("Dataset Scanning")

        self.scan_dataset()

        LOGGER.info(
            f"Found {len(self.images)} images."
        )

        return self.images
# ==========================================================
# Metadata Collector
# ==========================================================

class MetadataCollector:
    """
    Collect image metadata.

    Responsibilities
    ----------------
    1. Read image information
    2. Calculate MD5
    3. Calculate image size
    4. Calculate aspect ratio
    5. Calculate file size
    6. Detect blurry images

    Notes
    -----
    This class DOES NOT

    - validate labels
    - augment images
    - clean dataset
    """

    def __init__(
        self,
        config: PreprocessConfig,
    ) -> None:

        self.config = config

    # ------------------------------------------------------

    def collect(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:

        print_title("Collecting Metadata")

        LOGGER.info(
            f"Images: {len(images)}"
        )

        for image in tqdm(
            images,
            desc="Metadata",
        ):

            self.collect_single(image)

        LOGGER.info(
            "Metadata collection finished."
        )

        return images

    # ------------------------------------------------------

    def collect_single(
        self,
        image: ImageInfo,
    ) -> None:

        img = cv_imread(image.image_path)

        if img is None:

            image.is_valid = False

            image.error_message = (
                "Cannot read image."
            )

            return

        image.height = img.shape[0]

        image.width = img.shape[1]

        if len(img.shape) == 2:

            image.channels = 1

        else:

            image.channels = img.shape[2]

        image.file_size = (
            image.image_path.stat().st_size
        )

        image.aspect_ratio = round(

            image.width / image.height,

            4,

        )

        image.md5 = calculate_md5(

            image.image_path

        )

        image.image_format = (

            image.image_path.suffix.lower()

        )

        image.is_blurry = self.is_blurry(

            img

        )

    # ------------------------------------------------------

    def is_blurry(
        self,
        image: np.ndarray,
    ) -> bool:
        """
        Detect blurry image.

        Returns
        -------
        bool
        """

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY,

        )

        score = cv2.Laplacian(

            gray,

            cv2.CV_64F,

        ).var()

        return score < self.config.blur_threshold
    # ------------------------------------------------------

    def collect_statistics(
        self,
        images: List[ImageInfo],
    ) -> DatasetStatistics:
        """
        Collect dataset statistics.

        Parameters
        ----------
        images : List[ImageInfo]

        Returns
        -------
        DatasetStatistics
        """

        statistics = DatasetStatistics()

        for image in images:

            statistics.total_images += 1

            # ---------------- Label ---------------- #

            if image.has_label:

                statistics.labeled_images += 1

            else:

                statistics.unlabeled_images += 1

                statistics.missing_labels += 1

            # ---------------- Image Status ---------------- #

            if image.is_duplicate:

                statistics.duplicate_images += 1

            if image.is_blurry:

                statistics.blurry_images += 1

            if not image.is_valid:

                statistics.invalid_images += 1

            # ---------------- Bounding Box ---------------- #

            statistics.total_bboxes += image.bbox_count

            # ---------------- Class ---------------- #

            statistics.class_distribution.setdefault(
                image.class_name,
                0,
            )

            statistics.class_distribution[
                image.class_name
            ] += 1

            # ---------------- Image Format ---------------- #

            statistics.image_formats.setdefault(
                image.image_format,
                0,
            )

            statistics.image_formats[
                image.image_format
            ] += 1

            # ---------------- Resolution ---------------- #

            resolution = (
                f"{image.width}×{image.height}"
            )

            statistics.resolutions.setdefault(
                resolution,
                0,
            )

            statistics.resolutions[
                resolution
            ] += 1

        return statistics

    # ------------------------------------------------------

    def print_summary(
        self,
        statistics: DatasetStatistics,
    ) -> None:
        """
        Print dataset summary.
        """

        print_title("Dataset Summary")

        LOGGER.info(
            f"Total Images      : {statistics.total_images}"
        )

        LOGGER.info(
            f"Labeled Images    : {statistics.labeled_images}"
        )

        LOGGER.info(
            f"Missing Labels    : {statistics.missing_labels}"
        )

        LOGGER.info(
            f"Blurry Images     : {statistics.blurry_images}"
        )

        LOGGER.info(
            f"Duplicate Images  : {statistics.duplicate_images}"
        )

        LOGGER.info(
            f"Invalid Images    : {statistics.invalid_images}"
        )

        LOGGER.info(
            f"Total BoundingBox : {statistics.total_bboxes}"
        )

        LOGGER.info("")

        LOGGER.info("Class Distribution")

        for cls, num in statistics.class_distribution.items():

            LOGGER.info(
                f"  {cls:<12}: {num}"
            )

        LOGGER.info("")

        LOGGER.info("Image Formats")

        for fmt, num in statistics.image_formats.items():

            LOGGER.info(
                f"  {fmt:<8}: {num}"
            )

        LOGGER.info("")

        LOGGER.info("Image Resolution")

        for res, num in statistics.resolutions.items():

            LOGGER.info(
                f"  {res:<15}: {num}"
            )

    # ------------------------------------------------------

    def run(
        self,
        images: List[ImageInfo],
    ) -> Tuple[List[ImageInfo], DatasetStatistics]:
        """
        Execute metadata collection pipeline.

        Returns
        -------
        Tuple[
            List[ImageInfo],
            DatasetStatistics
        ]
        """

        images = self.collect(images)

        statistics = self.collect_statistics(
            images
        )

        self.print_summary(statistics)

        return images, statistics

# ==========================================================
# Label Loader
# ==========================================================

class LabelLoader:
    """
    Load YOLO label files.

    Responsibilities
    ----------------
    1. Read YOLO txt files
    2. Parse bounding boxes
    3. Save BoundingBox objects into ImageInfo
    4. Count bounding boxes

    Notes
    -----
    This class DOES NOT validate labels.
    Validation belongs to LabelValidator.
    """

    def __init__(
        self,
        config: PreprocessConfig,
    ) -> None:

        self.config = config

    # ------------------------------------------------------

    def load(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:

        print_title("Loading Labels")

        for image in tqdm(
            images,
            desc="Labels",
        ):

            self.load_single(image)

        LOGGER.info("Label loading finished.")

        return images

    # ------------------------------------------------------

    def load_single(
        self,
        image: ImageInfo,
    ) -> None:

        image.bboxes.clear()

        image.bbox_count = 0

        image.label_loaded = False

        if not image.has_label:

            return

        if image.label_path is None:

            return

        if not image.label_path.exists():

            image.label_error = "Label file not found."

            return

        try:

            with open(
                image.label_path,
                "r",
                encoding="utf-8",
            ) as f:

                lines = f.readlines()

        except Exception as e:

            image.label_error = str(e)

            return

        for line in lines:

            line = line.strip()

            if not line:

                continue

            parts = line.split()

            if len(parts) != 5:

                image.label_error = "Invalid label format."

                continue

            try:

                bbox = BoundingBox(

                    class_id=int(parts[0]),

                    x_center=float(parts[1]),

                    y_center=float(parts[2]),

                    width=float(parts[3]),

                    height=float(parts[4]),

                )

            except ValueError:

                image.label_error = "Cannot parse label."

                continue

            image.bboxes.append(bbox)

        image.bbox_count = len(image.bboxes)

        image.label_loaded = True

    # ------------------------------------------------------

    def print_summary(
        self,
        images: List[ImageInfo],
    ) -> None:

        total_labels = 0

        total_boxes = 0

        error_files = 0

        for image in images:

            if image.has_label:

                total_labels += 1

            total_boxes += image.bbox_count

            if image.label_error:

                error_files += 1

        print_title("Label Summary")

        LOGGER.info(
            f"Label Files : {total_labels}"
        )

        LOGGER.info(
            f"BoundingBoxes : {total_boxes}"
        )

        LOGGER.info(
            f"Error Labels : {error_files}"
        )

    # ------------------------------------------------------

    def run(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:

        images = self.load(images)

        self.print_summary(images)

        return images

# ==========================================================
# Image Cleaner
# ==========================================================

class ImageCleaner:
    """
    Clean dataset.

    Responsibilities
    ----------------
    1. Detect invalid images
    2. Detect duplicate images
    3. Detect empty labels
    4. Detect orphan labels
    5. Filter invalid samples

    Notes
    -----
    This class DOES NOT modify images.

    It only decides whether an image
    should be kept.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: PreprocessConfig,
    ) -> None:

        self.paths = paths

        self.config = config

    # ------------------------------------------------------

    def check_invalid_images(
        self,
        images: List[ImageInfo],
    ) -> None:

        LOGGER.info("Checking invalid images...")

        for image in images:

            if not image.is_valid:

                continue

            if image.width <= 0 or image.height <= 0:

                image.is_valid = False

                image.error_message = "Invalid image size."

    # ------------------------------------------------------

    def check_duplicate_images(
        self,
        images: List[ImageInfo],
    ) -> None:

        LOGGER.info("Checking duplicate images...")

        md5_table = {}

        for image in images:

            if image.md5 == "":

                continue

            if image.md5 in md5_table:

                image.is_duplicate = True

            else:

                md5_table[image.md5] = image

    # ------------------------------------------------------

    def check_empty_labels(
        self,
        images: List[ImageInfo],
    ) -> None:

        LOGGER.info("Checking empty labels...")

        for image in images:

            if not image.has_label:

                continue

            if image.bbox_count == 0:

                image.is_valid = False

                image.label_error = "Empty label."

    # ------------------------------------------------------

    def check_orphan_labels(self) -> int:

        LOGGER.info("Checking orphan labels...")

        orphan_count = 0

        image_names = {

            img.image_path.stem

            for img in self.paths.image_root.rglob("*")

            if img.is_file()

        }

        for txt in self.paths.label_root.glob("*.txt"):

            if txt.stem not in image_names:

                orphan_count += 1

        LOGGER.info(
            f"Orphan labels: {orphan_count}"
        )

        return orphan_count

    # ------------------------------------------------------

    def filter_images(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:

        LOGGER.info("Filtering dataset...")

        cleaned = []

        for image in images:

            if not image.is_valid:

                continue

            if image.is_duplicate:

                continue

            cleaned.append(image)

        LOGGER.info(
            f"Remain images: {len(cleaned)}"
        )

        return cleaned

    # ------------------------------------------------------

    def print_summary(
        self,
        images: List[ImageInfo],
    ) -> None:

        invalid = sum(
            1 for x in images
            if not x.is_valid
        )

        duplicate = sum(
            1 for x in images
            if x.is_duplicate
        )

        empty = sum(
            1 for x in images
            if x.label_error == "Empty label."
        )

        print_title("Image Cleaner Summary")

        LOGGER.info(
            f"Invalid Images : {invalid}"
        )

        LOGGER.info(
            f"Duplicate Images : {duplicate}"
        )

        LOGGER.info(
            f"Empty Labels : {empty}"
        )

    # ------------------------------------------------------

    def run(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:

        print_title("Image Cleaning")

        self.check_invalid_images(images)

        self.check_duplicate_images(images)

        self.check_empty_labels(images)

        self.check_orphan_labels()

        cleaned = self.filter_images(images)

        self.print_summary(cleaned)

        return cleaned

# ==========================================================
# Image Processor
# ==========================================================

class ImageProcessor:
    """
    Image preprocessing.

    Responsibilities
    ----------------
    1. LetterBox resize
    2. Save processed image
    3. Copy label file
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: PreprocessConfig,
    ) -> None:

        self.paths = paths
        self.config = config

    # ------------------------------------------------------

    def letterbox(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize image with unchanged aspect ratio.

        Similar to Ultralytics LetterBox.
        """

        target_h, target_w = self.config.target_size

        h, w = image.shape[:2]

        # ---------------- Scale ---------------- #

        ratio = min(
            target_w / w,
            target_h / h,
        )

        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=self.config.interpolation,
        )

        # ---------------- Padding ---------------- #

        dw = target_w - new_w
        dh = target_h - new_h

        left = dw // 2
        right = dw - left

        top = dh // 2
        bottom = dh - top

        padded = cv2.copyMakeBorder(

            resized,

            top,
            bottom,
            left,
            right,

            cv2.BORDER_CONSTANT,

            value=(114, 114, 114),

        )

        return padded

    # ------------------------------------------------------

    def save_image(
        self,
        image: np.ndarray,
        save_path: Path,
    ) -> None:
        """
        Save image.

        Parent folders are created automatically.
        """

        ensure_dir(save_path.parent)

        cv_imwrite(
            save_path,
            image,
        )

    # ------------------------------------------------------

    def copy_label(
        self,
        image_info: ImageInfo,
    ) -> None:
        """
        Copy YOLO label.

        Images without labels are skipped.
        """

        if not image_info.has_label:
            return

        if image_info.label_path is None:
            return

        if not image_info.label_path.exists():
            return

        ensure_dir(
            self.paths.output_label_root
        )

        shutil.copy2(

            image_info.label_path,

            self.paths.output_label_root
            / image_info.label_path.name,

        )

    # ------------------------------------------------------

    def build_output_path(
        self,
        image_info: ImageInfo,
    ) -> Path:
        """
        Build processed image path.

        Output filename remains unchanged.
        """

        return (

            self.paths.output_image_root

            / image_info.image_path.name

        )

    # ------------------------------------------------------

    def process_single(
        self,
        image_info: ImageInfo,
    ) -> bool:
        """
        Process one image.

        Returns
        -------
        bool
            True if successful.
        """

        img = cv_imread(image_info.image_path)

        if img is None:

            LOGGER.warning(
                f"Cannot read image: {image_info.image_path.name}"
            )

            return False

        # LetterBox Resize
        img = self.letterbox(img)

        # Output Path
        output_path = self.build_output_path(
            image_info
        )

        # Save Image
        self.save_image(
            img,
            output_path,
        )

        # Copy Label
        self.copy_label(
            image_info,
        )

        # 更新ImageInfo，使后续模块继续使用处理后的数据
        image_info.image_path = output_path

        if image_info.has_label:
            image_info.label_path = (

                    self.paths.output_label_root

                    / image_info.label_path.name
            )
        return True

    # ------------------------------------------------------

    def process_dataset(
        self,
        images: List[ImageInfo],
    ) -> Tuple[int, int]:
        """
        Process whole dataset.

        Returns
        -------
        success_count
        failed_count
        """

        success = 0
        failed = 0

        for image in tqdm(

            images,

            desc="Processing Images",

        ):

            if self.process_single(image):

                success += 1

            else:

                failed += 1

        return success, failed

    # ------------------------------------------------------

    def print_summary(
        self,
        success: int,
        failed: int,
    ) -> None:

        print_title(
            "Image Processing Summary"
        )

        LOGGER.info(
            f"Processed Images : {success}"
        )

        LOGGER.info(
            f"Failed Images    : {failed}"
        )

        LOGGER.info(
            f"Output Images    : {self.paths.output_image_root}"
        )

        LOGGER.info(
            f"Output Labels    : {self.paths.output_label_root}"
        )

    # ------------------------------------------------------

    def run(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:

        print_title(
            "Image Processing"
        )

        success, failed = self.process_dataset(
            images
        )

        self.print_summary(
            success,
            failed,
        )

        return images

# ==========================================================
# Data Augmentor
# ==========================================================

class DataAugmentor:
    """
    Data augmentation module.

    Workflow
    --------
    1. Read processed dataset
    2. Count samples of each class
    3. Calculate target number
    4. Augment minority classes
    5. Save augmented images and labels
    6. Return updated ImageInfo list
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: PreprocessConfig,
    ) -> None:

        self.paths = paths
        self.config = config

        self.transform = self.build_transform()

        self.total_generated = 0

        self.class_statistics: Dict[str, int] = {}

    # ------------------------------------------------------

    def build_transform(self) -> A.Compose:
        """
        Build Albumentations pipeline.

        Suitable for transmission-line foreign object detection.
        """

        return A.Compose(

            [

                A.Rotate(

                    limit=10,

                    border_mode=cv2.BORDER_CONSTANT,

                    fill=(114, 114, 114),

                    p=0.4,

                ),

                A.HorizontalFlip(

                    p=0.5,

                ),

                A.RandomBrightnessContrast(

                    brightness_limit=0.15,

                    contrast_limit=0.15,

                    p=0.4,

                ),

                A.ColorJitter(

                    brightness=0.15,

                    contrast=0.15,

                    saturation=0.15,

                    hue=0.05,

                    p=0.3,

                ),

                A.CLAHE(

                    clip_limit=2.0,

                    tile_grid_size=(8, 8),

                    p=0.2,

                ),

                A.MotionBlur(

                    blur_limit=5,

                    p=0.15,

                ),

                A.GaussNoise(

                    std_range=(0.02, 0.08),

                    p=0.15,

                ),

            ],

            bbox_params=A.BboxParams(

                format="yolo",

                label_fields=["class_labels"],

                min_visibility=0.1,

            ),

        )

    # ------------------------------------------------------

    def count_classes(
        self,
        images: List[ImageInfo],
    ) -> Dict[str, int]:
        """
        Count labeled samples for each class.
        """

        counter: Dict[str, int] = {}

        for info in images:

            if not info.has_label:
                continue

            if info.bbox_count == 0:
                continue

            counter[info.class_name] = (

                counter.get(info.class_name, 0) + 1

            )

        self.class_statistics = counter

        return counter

    # ------------------------------------------------------

    def calculate_target(
        self,
        counter: Dict[str, int],
    ) -> int:
        """
        Calculate target sample number.

        The largest class is used as the target.
        """

        if len(counter) == 0:

            return 0

        return max(counter.values())

    # ------------------------------------------------------

    def save_augmented_sample(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        source_info: ImageInfo,
        index: int,
    ) -> ImageInfo:
        """
        Save augmented image and label, then build ImageInfo.
        """

        image_name = (
            f"{source_info.image_path.stem}_aug_{index}"
            f"{source_info.image_path.suffix}"
        )

        label_name = (
            f"{source_info.image_path.stem}_aug_{index}.txt"
        )

        image_path = (
            self.paths.output_image_root / image_name
        )

        label_path = (
            self.paths.output_label_root / label_name
        )

        ensure_dir(image_path.parent)
        ensure_dir(label_path.parent)

        cv_imwrite(image_path, image)

        with open(label_path, "w", encoding="utf-8") as f:

            for cls, box in zip(class_labels, bboxes):

                f.write(
                    f"{cls} "
                    f"{box[0]:.6f} "
                    f"{box[1]:.6f} "
                    f"{box[2]:.6f} "
                    f"{box[3]:.6f}\n"
                )

        new_info = ImageInfo(

            image_path=image_path,

            label_path=label_path,

            class_name=source_info.class_name,

            has_label=True,

            image_format=image_path.suffix.lower(),

        )

        new_info.width = image.shape[1]
        new_info.height = image.shape[0]
        new_info.channels = image.shape[2]

        new_info.bbox_count = len(bboxes)

        new_info.bboxes = [

            BoundingBox(

                class_id=cls,

                x_center=box[0],

                y_center=box[1],

                width=box[2],

                height=box[3],

            )

            for cls, box in zip(class_labels, bboxes)

        ]

        return new_info

    # ------------------------------------------------------

    def augment_image(
        self,
        image_info: ImageInfo,
        index: int,
    ) -> Optional[ImageInfo]:
        """
        Augment one image.

        Returns
        -------
        ImageInfo | None
        """

        if not image_info.has_label:
            return None

        if image_info.bbox_count == 0:
            return None

        image = cv_imread(image_info.image_path)

        if image is None:

            LOGGER.warning(
                f"Cannot read image: "
                f"{image_info.image_path.name}"
            )

            return None

        bboxes = []

        class_labels = []

        for bbox in image_info.bboxes:

            bboxes.append(

                [

                    bbox.x_center,

                    bbox.y_center,

                    bbox.width,

                    bbox.height,

                ]

            )

            class_labels.append(

                bbox.class_id

            )

        try:

            result = self.transform(

                image=image,

                bboxes=bboxes,

                class_labels=class_labels,

            )

        except Exception as e:

            LOGGER.warning(str(e))

            return None

        if len(result["bboxes"]) == 0:

            return None

        self.total_generated += 1

        return self.save_augmented_sample(

            image=result["image"],

            bboxes=result["bboxes"],

            class_labels=result["class_labels"],

            source_info=image_info,

            index=index,

        )

    # ------------------------------------------------------

    def augment_class(
        self,
        images: List[ImageInfo],
        class_name: str,
        target_count: int,
    ) -> List[ImageInfo]:
        """
        Augment one class until it reaches the target count.

        Returns
        -------
        List[ImageInfo]
            Newly generated samples.
        """

        candidates = [

            image

            for image in images

            if (
                image.class_name == class_name
                and image.has_label
                and image.bbox_count > 0
            )

        ]

        current_count = len(candidates)

        if current_count == 0:

            LOGGER.warning(
                f"No labeled samples found for class '{class_name}'."
            )

            return []

        if current_count >= target_count:

            LOGGER.info(
                f"{class_name:<12}: "
                f"{current_count} (Already Balanced)"
            )

            return []

        need = target_count - current_count

        LOGGER.info(

            f"{class_name:<12}: "

            f"{current_count} -> {target_count} "

            f"(Generate {need})"

        )

        generated_images: List[ImageInfo] = []

        attempt = 0

        max_attempt = need * 5

        while (

            len(generated_images) < need

            and attempt < max_attempt

        ):

            attempt += 1

            source = random.choice(candidates)

            new_info = self.augment_image(

                source,

                len(generated_images),

            )

            if new_info is None:

                continue

            generated_images.append(new_info)

        if len(generated_images) < need:

            LOGGER.warning(

                f"{class_name}: "

                f"Only generated "

                f"{len(generated_images)}/{need} samples."

            )

        return generated_images

    # ------------------------------------------------------

    def print_summary(
        self,
        original_count: int,
        generated_count: int,
        final_count: int,
    ) -> None:
        """
        Print augmentation summary.
        """

        print_title("Augmentation Summary")

        LOGGER.info(
            f"Original Images : {original_count}"
        )

        LOGGER.info(
            f"Generated Images: {generated_count}"
        )

        LOGGER.info(
            f"Final Images    : {final_count}"
        )

        LOGGER.info("")

        LOGGER.info("Class Distribution:")

        for cls, num in self.class_statistics.items():

            LOGGER.info(

                f"{cls:<12}: {num}"

            )

    # ------------------------------------------------------

    def run(
        self,
        images: List[ImageInfo],
    ) -> List[ImageInfo]:
        """
        Execute the complete data augmentation pipeline.

        Parameters
        ----------
        images : List[ImageInfo]
            Current processed dataset.

        Returns
        -------
        List[ImageInfo]
            Dataset including newly augmented samples.
        """

        print_title("Data Augmentation")

        original_count = len(images)

        class_counter = self.count_classes(images)

        if len(class_counter) == 0:

            LOGGER.warning(
                "No labeled images found, skip augmentation."
            )

            return images

        LOGGER.info("Current class distribution:")

        for cls, num in class_counter.items():

            LOGGER.info(f"{cls:<12}: {num}")

        target_count = self.calculate_target(class_counter)

        LOGGER.info("")
        LOGGER.info(
            f"Target samples per class: {target_count}"
        )

        new_images: List[ImageInfo] = []

        for class_name in sorted(class_counter.keys()):

            generated = self.augment_class(

                images,

                class_name,

                target_count,

            )

            new_images.extend(generated)

        images.extend(new_images)

        generated_count = len(new_images)

        final_count = len(images)

        # 更新最终类别统计
        final_counter = self.count_classes(images)

        self.class_statistics = final_counter

        self.print_summary(

            original_count,

            generated_count,

            final_count,

        )

        LOGGER.info("")
        LOGGER.info("Final Class Distribution:")

        for cls, num in final_counter.items():

            LOGGER.info(f"{cls:<12}: {num}")

        LOGGER.info("")
        LOGGER.info("Data augmentation completed.")

        return images

# ==========================================================
# Dataset Splitter
# ==========================================================

class DatasetSplitter:
    """
    Split processed dataset into train / validation dataset.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: PreprocessConfig,
    ) -> None:

        self.paths = paths
        self.config = config

        # output folders

        self.train_image_dir = (

            self.paths.output_image_root

            / "train"

        )

        self.val_image_dir = (

            self.paths.output_image_root

            / "val"

        )

        self.train_label_dir = (

            self.paths.output_label_root

            / "train"

        )

        self.val_label_dir = (

            self.paths.output_label_root

            / "val"

        )

        ensure_dir(self.train_image_dir)

        ensure_dir(self.val_image_dir)

        ensure_dir(self.train_label_dir)

        ensure_dir(self.val_label_dir)

    # ------------------------------------------------------

    def split_dataset(
        self,
        images: List[ImageInfo],
    ) -> Tuple[List[ImageInfo], List[ImageInfo]]:
        """
        Split dataset into training and validation subsets
        using stratified sampling.
        """

        if len(images) == 0:

            return [], []

        labels = [

            image.class_name

            for image in images

        ]

        train_images, val_images = train_test_split(

            images,

            train_size=self.config.train_ratio,

            random_state=self.config.random_seed,

            shuffle=True,

            stratify=labels,

        )

        LOGGER.info(

            f"Train Samples : {len(train_images)}"

        )

        LOGGER.info(

            f"Validation Samples : {len(val_images)}"

        )

        return train_images, val_images

    # ------------------------------------------------------

    def copy_sample(
        self,
        image_info: ImageInfo,
        subset: str,
    ) -> None:
        """
        Copy one image-label pair to processed dataset.

        Parameters
        ----------
        subset
            "train" or "val"
        """

        if subset == "train":

            image_dst = (

                self.train_image_dir

                / image_info.image_path.name

            )

            label_dst = (

                self.train_label_dir

                / image_info.label_path.name

            )

        else:

            image_dst = (

                self.val_image_dir

                / image_info.image_path.name

            )

            label_dst = (

                self.val_label_dir

                / image_info.label_path.name

            )

        shutil.copy2(

            image_info.image_path,

            image_dst,

        )

        if image_info.has_label:

            shutil.copy2(

                image_info.label_path,

                label_dst,

            )

        image_info.image_path = image_dst

        if image_info.has_label:
            image_info.label_path = label_dst

    # ------------------------------------------------------

    def run(
        self,
        images: List[ImageInfo],
    ) -> Tuple[List[ImageInfo], List[ImageInfo]]:
        """
        Split processed dataset into train / validation dataset.

        Returns
        -------
        train_images
            Training dataset.

        val_images
            Validation dataset.
        """

        print_title("Dataset Split")

        train_images, val_images = self.split_dataset(
            images
        )

        LOGGER.info("")
        LOGGER.info("Copying training dataset...")

        for image in tqdm(
            train_images,
            desc="Train",
            unit="image",
        ):

            self.copy_sample(
                image,
                "train",
            )

        LOGGER.info("")
        LOGGER.info("Copying validation dataset...")

        for image in tqdm(
            val_images,
            desc="Validation",
            unit="image",
        ):

            self.copy_sample(
                image,
                "val",
            )

        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("Dataset Split Completed")
        LOGGER.info("=" * 60)

        LOGGER.info(
            f"Training Images   : {len(train_images)}"
        )

        LOGGER.info(
            f"Validation Images : {len(val_images)}"
        )

        LOGGER.info(
            f"Total Images      : {len(images)}"
        )

        LOGGER.info("")

        return train_images, val_images

# ==========================================================
# Report Generator
# ==========================================================

class ReportGenerator:
    """
    Generate preprocessing reports.

    Output
    ------
    1. dataset_statistics.json
    2. preprocess_report.md
    """

    def __init__(
        self,
        paths: ProjectPaths,
    ) -> None:

        self.paths = paths

        ensure_dir(
            self.paths.report_root
        )

    # ------------------------------------------------------

    def build_statistics(
            self,
            train_images: List[ImageInfo],
            val_images: List[ImageInfo],
    ) -> Dict[str, Any]:
        """
        Build dataset statistics.
        """

        statistics = {

            "train_images": len(train_images),

            "validation_images": len(val_images),

            "total_images": (

                len(train_images)

                + len(val_images)

            ),

            "classes": {},

            "generated_time": datetime.now(

            ).strftime(

                "%Y-%m-%d %H:%M:%S"

            ),

        }

        class_counter = {}

        for image in train_images + val_images:

            class_counter[

                image.class_name

            ] = (

                class_counter.get(

                    image.class_name,

                    0,

                )

                + 1

            )

        statistics["classes"] = class_counter

        return statistics

    # ------------------------------------------------------

    def export_json(
        self,
        statistics: Dict[str, Any],
    ) -> Path:
        """
        Export dataset statistics to JSON.
        """

        json_path = (
            self.paths.report_root
            / "dataset_statistics.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(

                statistics,

                f,

                indent=4,

                ensure_ascii=False,

            )

        LOGGER.info(
            f"JSON report saved: {json_path.name}"
        )

        return json_path

    # ------------------------------------------------------

    def export_markdown(
        self,
        statistics: Dict[str, Any],
    ) -> Path:
        """
        Export preprocessing report to Markdown.
        """

        md_path = (
            self.paths.report_root
            / "preprocess_report.md"
        )

        lines = []

        lines.append("# Dataset Preprocessing Report\n")

        lines.append(
            f"**Generated Time:** "
            f"{statistics['generated_time']}\n"
        )

        lines.append("## Dataset Summary\n")

        lines.append(
            f"- Total Images: "
            f"{statistics['total_images']}\n"
        )

        lines.append(
            f"- Training Images: "
            f"{statistics['train_images']}\n"
        )

        lines.append(
            f"- Validation Images: "
            f"{statistics['validation_images']}\n"
        )

        lines.append("\n## Class Distribution\n")

        lines.append("| Class | Images |\n")
        lines.append("|------|-------:|\n")

        for cls, num in statistics["classes"].items():

            lines.append(
                f"| {cls} | {num} |\n"
            )

        lines.append("\n---\n")

        lines.append(
            "This report was automatically generated "
            "by preprocess.py.\n"
        )

        with open(
            md_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.writelines(lines)

        LOGGER.info(
            f"Markdown report saved: {md_path.name}"
        )

        return md_path

    # ------------------------------------------------------

    def run(
        self,
        train_images: List[ImageInfo],
        val_images: List[ImageInfo],
    ) -> None:
        """
        Generate all preprocessing reports.
        """

        print_title("Generate Report")

        statistics = self.build_statistics(
            train_images,
            val_images,
        )

        self.export_json(statistics)

        self.export_markdown(statistics)

        LOGGER.info(
            "Report generation completed."
        )

def main():

    config = PreprocessConfig()

    paths = ProjectPaths()

    scanner = DatasetScanner(paths, config)
    images = scanner.run()

    cleaner = ImageCleaner(paths, config)
    images = cleaner.run(images)

    processor = ImageProcessor(paths, config)
    images = processor.run(images)

    augmentor = DataAugmentor(paths, config)
    images = augmentor.run(images)

    splitter = DatasetSplitter(paths, config)
    train_images, val_images = splitter.run(images)

    reporter = ReportGenerator(paths)
    reporter.run(train_images, val_images)

    LOGGER.info("Preprocessing completed.")

if __name__ == "__main__":
    main()