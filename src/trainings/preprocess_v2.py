"""
preprocess_v2.py

Enterprise Dataset Preprocessing Pipeline
Author : ChatGPT
Project: Pole_AI_Project

Functions
---------
1. Scan dataset
2. Clean invalid samples
3. Process images
4. Data augmentation
5. Split dataset
6. Generate reports
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import hashlib
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import List, Tuple
from collections import Counter

@dataclass
class PreprocessConfig:
    """
    数据预处理配置

    所有模块均从此配置读取参数，
    不允许在业务代码中出现硬编码。
    """

    # ==========================
    # 数据集配置
    # ==========================

    class_names: List[str] = field(default_factory=lambda: [
        "其他异物",
        "塑料袋",
        "气球",
        "鸟巢"
    ])

    image_suffixes: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    label_suffix: str = ".txt"

    # ==========================
    # 图像处理配置
    # ==========================

    target_size: int = 640

    keep_ratio: bool = True

    interpolation: int = cv2.INTER_LINEAR

    # ==========================
    # 数据划分
    # ==========================

    train_ratio: float = 0.8

    random_seed: int = 42

    stratified_split: bool = True

    # ==========================
    # 数据增强
    # ==========================

    enable_augmentation: bool = True

    augmentation_target: int = 60
    """
    每个类别最少扩充到多少张。

    若某类别已有超过该数量，则不会继续增强。
    """

    # ==========================
    # 清洗配置
    # ==========================

    remove_invalid_images: bool = True

    remove_empty_labels: bool = True

    remove_orphan_labels: bool = True

    remove_duplicate_images: bool = True

    # ==========================
    # 日志配置
    # ==========================

    log_level: str = "INFO"

    show_progress: bool = True

    # ==========================
    # 输出配置
    # ==========================

    save_statistics_json: bool = True

    save_markdown_report: bool = True

    overwrite_processed: bool = True

@dataclass
class ProjectPaths:
    """
    项目路径管理

    统一管理整个项目所有路径。

    所有模块只能从这里获取路径，
    禁止自行拼接目录。
    """

    config: PreprocessConfig

    def __post_init__(self):

        # 项目根目录
        self.project_root = Path(__file__).resolve().parent.parent

        # 数据目录
        self.data_dir = self.project_root / "data"

        # 原始YOLO数据集（只读）
        self.dataset_dir = self.data_dir / "yolo_dataset"

        self.images_root = self.dataset_dir / "images"
        self.labels_root = self.dataset_dir / "labels"

        self.images_train = self.images_root / "train"
        self.images_val = self.images_root / "val"

        self.labels_train = self.labels_root / "train"
        self.labels_val = self.labels_root / "val"

        # ==========================
        # processed
        # ==========================

        self.processed_dir = self.data_dir / "processed"

        self.processed_images = self.processed_dir / "images"
        self.processed_labels = self.processed_dir / "labels"

        # 数据集划分目录
        self.split_dir = self.processed_dir / "split"

        self.split_images = self.split_dir / "images"
        self.split_labels = self.split_dir / "labels"

        self.train_images = self.split_images / "train"
        self.val_images = self.split_images / "val"

        self.train_labels = self.split_labels / "train"
        self.val_labels = self.split_labels / "val"

        self.report_dir = self.processed_dir / "reports"

        self.visualization_dir = self.processed_dir / "visualization"

        self.cache_dir = self.processed_dir / "cache"

    def create_directories(self) -> None:
        """
        创建所有输出目录。

        原始数据目录不会创建，
        只创建processed目录。
        """

        directories = [

            self.processed_dir,

            self.processed_images,
            self.processed_labels,

            self.split_dir,
            self.split_images,
            self.split_labels,

            self.train_images,
            self.val_images,

            self.train_labels,
            self.val_labels,

            self.report_dir,

            self.visualization_dir,

            self.cache_dir

        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def verify_dataset(self) -> None:
        """
        检查原始数据集是否存在。
        """

        if not self.images_train.exists():
            raise FileNotFoundError(
                f"Images directory not found:\n{self.images_dir}"
            )

        if not self.labels_train.exists():
            raise FileNotFoundError(
                f"Labels directory not found:\n{self.labels_dir}"
            )

    def initialize(self) -> None:
        """
        初始化项目目录。

        包括：

        1. 检查原始数据集
        2. 创建processed目录
        """

        self.verify_dataset()

        self.create_directories()

        logging.info("Project directories initialized successfully.")

# ==========================================================
# Data Classes
# ==========================================================

@dataclass
class BoundingBox:
    """
    YOLO格式目标框

    Attributes
    ----------
    class_id : int
        类别编号

    x_center : float
        中心点x（归一化）

    y_center : float
        中心点y（归一化）

    width : float
        宽（归一化）

    height : float
        高（归一化）
    """

    class_id: int

    x_center: float

    y_center: float

    width: float

    height: float


@dataclass
class ImageInfo:
    """
    单张图片信息

    整个预处理流程都围绕该对象进行，
    各模块通过修改对象属性传递处理结果。
    """

    # 文件信息
    image_path: Path
    label_path: Optional[Path] = None

    file_name: str = ""

    # 图像属性
    width: int = 0
    height: int = 0
    channels: int = 3

    # 标签
    bboxes: List[BoundingBox] = field(default_factory=list)

    # 数据集类别（用于分层划分）
    primary_class: Optional[int] = None

    # 状态标记
    is_valid: bool = True

    is_processed: bool = False

    is_augmented: bool = False

    # 数据来源
    source_image: Optional[Path] = None

    # 备注
    message: str = ""


@dataclass
class DatasetStatistics:
    """
    数据集统计信息
    """

    total_images: int = 0

    valid_images: int = 0

    invalid_images: int = 0

    total_labels: int = 0

    total_boxes: int = 0

    class_distribution: Counter = field(default_factory=Counter)

    image_formats: Counter = field(default_factory=Counter)

    image_sizes: Counter = field(default_factory=Counter)

    duplicate_images: int = 0

    augmented_images: int = 0

    train_images: int = 0

    val_images: int = 0

# ==========================================================
# Dataset Scanner
# ==========================================================

class DatasetScanner:
    """
    扫描YOLO数据集，构建ImageInfo对象。
    """

    def __init__(
        self,
        config: PreprocessConfig,
        paths: ProjectPaths
    ):

        self.config = config
        self.paths = paths

    def _scan_images(self) -> List[Path]:
        """
        递归扫描所有图片。
        """

        image_files = []

        for suffix in self.config.image_suffixes:
            image_files.extend(
                self.paths.train_images.rglob(f"*{suffix}")
            )

        image_files.sort()

        logging.info(
            "Found %d images.",
            len(image_files)
        )

        return image_files

    def _find_label(
        self,
        image_path: Path
    ) -> Optional[Path]:
        """
        根据图片查找对应标签。
        """

        label_path = (
                self.paths.train_labels /
                image_path.with_suffix(".txt").name
        )

        if label_path.exists():
            return label_path

        return None

    def _load_yolo_label(
        self,
        label_path: Optional[Path]
    ) -> List[BoundingBox]:
        """
        读取YOLO标签。
        """

        boxes = []

        if label_path is None:
            return boxes

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    boxes.append(

                        BoundingBox(

                            class_id=int(values[0]),

                            x_center=float(values[1]),

                            y_center=float(values[2]),

                            width=float(values[3]),

                            height=float(values[4])

                        )

                    )

        except Exception as e:

            logging.warning(
                "Failed to read label: %s (%s)",
                label_path,
                e
            )

        return boxes

    def _read_image_info(
        self,
        image_path: Path
    ) -> ImageInfo:
        """
        构建ImageInfo对象。
        """

        image = cv2.imread(str(image_path))

        if image is None:

            return ImageInfo(

                image_path=image_path,

                file_name=image_path.name,

                is_valid=False,

                message="Failed to load image."

            )

        height, width = image.shape[:2]

        channels = 1 if len(image.shape) == 2 else image.shape[2]

        label_path = self._find_label(image_path)

        boxes = self._load_yolo_label(label_path)

        primary_class = None

        if boxes:
            primary_class = boxes[0].class_id

        return ImageInfo(

            image_path=image_path,

            label_path=label_path,

            file_name=image_path.name,

            width=width,

            height=height,

            channels=channels,

            bboxes=boxes,

            primary_class=primary_class

        )

    def scan(
        self
    ) -> List[ImageInfo]:
        """
        扫描整个数据集。
        """

        image_infos = []

        image_files = self._scan_images()

        iterator = tqdm(
            image_files,
            desc="Scanning Dataset",
            disable=not self.config.show_progress
        )

        for image_path in iterator:

            image_infos.append(

                self._read_image_info(
                    image_path
                )

            )

        logging.info(
            "Dataset scan completed."
        )

        return image_infos

    def collect_statistics(
        self,
        image_infos: List[ImageInfo]
    ) -> DatasetStatistics:
        """
        根据扫描结果统计数据集信息。
        """

        stats = DatasetStatistics()

        stats.total_images = len(image_infos)

        target_classes = self._get_target_classes(image_infos)

        for image_info in image_infos:

            if image_info.primary_class not in target_classes:
                continue

            if image_info.is_valid:
                stats.valid_images += 1
            else:
                stats.invalid_images += 1

            suffix = image_info.image_path.suffix.lower()
            stats.image_formats[suffix] += 1

            size = (
                image_info.width,
                image_info.height
            )

            stats.image_sizes[size] += 1

            if image_info.label_path is not None:
                stats.total_labels += 1

            for bbox in image_info.bboxes:

                stats.total_boxes += 1

                stats.class_distribution[
                    bbox.class_id
                ] += 1

        return stats

    def print_statistics(
        self,
        stats: DatasetStatistics
    ) -> None:
        """
        打印数据集统计信息。
        """

        logging.info("=" * 50)
        logging.info("Dataset Statistics")
        logging.info("=" * 50)

        logging.info(
            "Total Images      : %d",
            stats.total_images
        )

        logging.info(
            "Valid Images      : %d",
            stats.valid_images
        )

        logging.info(
            "Invalid Images    : %d",
            stats.invalid_images
        )

        logging.info(
            "Total Labels      : %d",
            stats.total_labels
        )

        logging.info(
            "Total Objects     : %d",
            stats.total_boxes
        )

        logging.info("-" * 50)

        logging.info("Class Distribution")

        for class_id, count in sorted(
            stats.class_distribution.items()
        ):

            class_name = self.config.class_names[class_id]

            logging.info(
                "%s : %d",
                class_name,
                count
            )

        logging.info("-" * 50)

        logging.info("Image Formats")

        for suffix, count in sorted(
            stats.image_formats.items()
        ):

            logging.info(
                "%s : %d",
                suffix,
                count
            )

        logging.info("=" * 50)

    def scan_dataset(
        self
    ) -> Tuple[List[ImageInfo], DatasetStatistics]:
        """
        扫描数据集并完成统计。

        Returns
        -------
        image_infos :
            所有图片信息

        stats :
            数据集统计信息
        """

        image_infos = self.scan()

        stats = self.collect_statistics(
            image_infos
        )

        self.print_statistics(
            stats
        )

        return image_infos, stats

# ==========================================================
# Image Cleaner
# ==========================================================

class ImageCleaner:
    """
    数据清洗模块

    功能：
    1. 检查损坏图片
    2. 检查空标签
    3. 检查重复图片
    4. 返回有效样本
    """

    def __init__(self, config: PreprocessConfig):

        self.config = config

    @staticmethod
    def _calculate_md5(image_path: Path) -> str:
        """
        计算图片MD5。
        """

        md5 = hashlib.md5()

        with open(image_path, "rb") as f:

            for chunk in iter(lambda: f.read(4096), b""):

                md5.update(chunk)

        return md5.hexdigest()

    def clean(
        self,
        image_infos: List[ImageInfo]
    ) -> List[ImageInfo]:
        """
        清洗数据集。

        Parameters
        ----------
        image_infos
            扫描后的图片信息

        Returns
        -------
        List[ImageInfo]
            清洗后的图片列表
        """

        cleaned_infos = []

        md5_cache = {}

        target_classes = self._get_target_classes(image_infos)

        for image_info in tqdm(
                image_infos,
                desc="Augmenting",
                disable=not self.config.show_progress
        ):

            if image_info.primary_class not in target_classes:
                continue

            # -------------------------
            # 已失效图片
            # -------------------------
            if not image_info.is_valid:

                continue

            # -------------------------
            # 检查空Label
            # -------------------------
            if self.config.remove_empty_labels:

                if image_info.label_path is None:

                    image_info.is_valid = False
                    image_info.message = "Missing label"

                    continue

                if image_info.label_path.stat().st_size == 0:

                    image_info.is_valid = False
                    image_info.message = "Empty label"

                    continue

            # -------------------------
            # 检查重复图片
            # -------------------------
            if self.config.remove_duplicate_images:

                image_md5 = self._calculate_md5(
                    image_info.image_path
                )

                if image_md5 in md5_cache:

                    image_info.is_valid = False
                    image_info.message = (
                        "Duplicate image"
                    )

                    continue

                md5_cache[image_md5] = (
                    image_info.image_path
                )

            cleaned_infos.append(image_info)

        logging.info(
            "Image Cleaning Finished"
        )

        logging.info(
            "Before Cleaning : %d",
            len(image_infos)
        )

        logging.info(
            "After Cleaning  : %d",
            len(cleaned_infos)
        )

        logging.info(
            "Removed         : %d",
            len(image_infos) - len(cleaned_infos)
        )

        return cleaned_infos

# ==========================================================
# Image Processor
# ==========================================================

class ImageProcessor:
    """
    图像预处理模块

    功能：
    1. Resize / LetterBox
    2. 保存处理后的图片
    3. 复制对应Label
    4. 更新ImageInfo路径
    """

    def __init__(
        self,
        config: PreprocessConfig,
        paths: ProjectPaths
    ):

        self.config = config
        self.paths = paths

        self.paths.processed_images.mkdir(
            parents=True,
            exist_ok=True
        )

        self.paths.processed_labels.mkdir(
            parents=True,
            exist_ok=True
        )

    def _letterbox(
            self,
            image: np.ndarray
    ) -> np.ndarray:
        """
        等比例缩放并补边（YOLO LetterBox）。

        Parameters
        ----------
        image
            原始图片

        Returns
        -------
        np.ndarray
            LetterBox后的图片
        """

        target = self.config.target_size

        h, w = image.shape[:2]

        scale = min(
            target / h,
            target / w
        )

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=self.config.interpolation
        )

        canvas = np.full(
            (
                target,
                target,
                3
            ),
            114,
            dtype=np.uint8
        )

        x_offset = (target - new_w) // 2
        y_offset = (target - new_h) // 2

        canvas[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = resized

        return canvas

    def process(
        self,
        image_infos: List[ImageInfo]
    ) -> List[ImageInfo]:

        processed_infos = []

        for image_info in tqdm(
            image_infos,
            desc="Processing Images",
            disable=not self.config.show_progress
        ):

            image = cv2.imread(
                str(image_info.image_path)
            )

            if image is None:

                logging.warning(
                    "Failed to read image: %s",
                    image_info.image_path
                )

                continue

            image = self._letterbox(image)

            output_image = (
                self.paths.processed_images /
                image_info.image_path.name
            )

            success = cv2.imwrite(
                str(output_image),
                image
            )

            if not success:
                logging.warning(
                    "Failed to save image: %s",
                    output_image
                )
                continue

            output_label = (
                self.paths.processed_labels /
                image_info.label_path.name
            )

            if not image_info.label_path.exists():
                logging.warning(
                    "Missing label: %s",
                    image_info.label_path
                )
                continue

            shutil.copy2(
                image_info.label_path,
                output_label
            )

            image_info.image_path = output_image
            image_info.label_path = output_label

            processed_infos.append(
                image_info
            )

        logging.info(
            "Image Processing Finished"
        )

        logging.info(
            "Processed Images : %d",
            len(processed_infos)
        )

        return processed_infos

# ==========================================================
# Data Augmentor
# ==========================================================

class DataAugmentor:
    """
    数据增强模块

    功能：
    1. 水平翻转
    2. 亮度增强
    3. 保存增强图片
    4. 复制Label
    """

    def __init__(
        self,
        config: PreprocessConfig,
        paths: ProjectPaths
    ):

        self.config = config
        self.paths = paths

    @staticmethod
    def _flip(image):

        return cv2.flip(image, 1)

    @staticmethod
    def _brightness(
        image,
        alpha=1.1,
        beta=15
    ):

        return cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=beta
        )

    def _get_target_classes(
            self,
            image_infos: List[ImageInfo]
    ) -> set[int]:
        """
        获取需要进行数据增强的类别。
        """

        class_counter = Counter()

        for image_info in image_infos:

            if image_info.primary_class is None:
                continue

            class_counter[image_info.primary_class] += 1

        if not class_counter:
            return set()

        max_count = max(class_counter.values())

        target_classes = set()

        for class_id, count in class_counter.items():

            if count < max_count:
                target_classes.add(class_id)

        return target_classes

    def _save_augmented(
        self,
        image,
        image_info: ImageInfo,
        suffix: str
    ) -> ImageInfo:

        image_name = (
            image_info.image_path.stem +
            suffix +
            image_info.image_path.suffix
        )

        label_name = (
            image_info.label_path.stem +
            suffix +
            image_info.label_path.suffix
        )

        image_path = (
            self.paths.processed_images /
            image_name
        )

        label_path = (
            self.paths.processed_labels /
            label_name
        )

        cv2.imwrite(
            str(image_path),
            image
        )

        shutil.copy2(
            image_info.label_path,
            label_path
        )

        return ImageInfo(
            image_path=image_path,
            label_path=label_path,
            file_name=image_name,
            width=image.shape[1],
            height=image.shape[0],
            bboxes=[
                BoundingBox(
                    class_id=box.class_id,
                    x_center=box.x_center,
                    y_center=box.y_center,
                    width=box.width,
                    height=box.height
                )
                for box in image_info.bboxes
            ],
            is_valid=True
        )

    def augment(
        self,
        image_infos: List[ImageInfo]
    ) -> List[ImageInfo]:

        augmented_infos = image_infos.copy()

        for image_info in tqdm(
            image_infos,
            desc="Augmenting",
            disable=not self.config.show_progress
        ):

            image = cv2.imread(
                str(image_info.image_path)
            )

            if image is None:

                continue



            # Brightness
            bright_image = self._brightness(image)

            augmented_infos.append(

                self._save_augmented(
                    bright_image,
                    image_info,
                    "_bright"
                )

            )

        logging.info(
            "Augmented Images : %d",
            len(augmented_infos)
        )

        return augmented_infos

# ==========================================================
# Dataset Splitter
# ==========================================================

class DatasetSplitter:
    """
    数据集划分模块

    功能：
    1. 按类别划分 Train / Validation
    2. 复制图片与标签
    3. 更新 ImageInfo 路径
    """

    def __init__(
        self,
        config: PreprocessConfig,
        paths: ProjectPaths
    ):

        self.config = config
        self.paths = paths

        self.paths.train_images.mkdir(
            parents=True,
            exist_ok=True
        )

        self.paths.val_images.mkdir(
            parents=True,
            exist_ok=True
        )

        self.paths.train_labels.mkdir(
            parents=True,
            exist_ok=True
        )

        self.paths.val_labels.mkdir(
            parents=True,
            exist_ok=True
        )

    def _copy_sample(
        self,
        image_info: ImageInfo,
        image_dir: Path,
        label_dir: Path
    ) -> ImageInfo:

        new_image = image_dir / image_info.image_path.name
        new_label = label_dir / image_info.label_path.name

        shutil.copy2(
            image_info.image_path,
            new_image
        )

        shutil.copy2(
            image_info.label_path,
            new_label
        )

        new_info = copy.deepcopy(image_info)

        new_info.image_path = new_image
        new_info.label_path = new_label

        return new_info

    def split(
        self,
        image_infos: List[ImageInfo]
    ) -> tuple[list[ImageInfo], list[ImageInfo]]:

        labels = [

            image.primary_class

            for image in image_infos

        ]

        train_infos, val_infos = train_test_split(

            image_infos,

            test_size=self.config.val_ratio,

            random_state=self.config.random_seed,

            shuffle=True,

            stratify=labels

        )

        train_results = []

        val_results = []

        for image_info in tqdm(

            train_infos,

            desc="Copy Train",

            disable=not self.config.show_progress

        ):

            train_results.append(

                self._copy_sample(

                    image_info,

                    self.paths.train_images,

                    self.paths.train_labels

                )

            )

        for image_info in tqdm(

            val_infos,

            desc="Copy Validation",

            disable=not self.config.show_progress

        ):

            val_results.append(

                self._copy_sample(

                    image_info,

                    self.paths.val_images,

                    self.paths.val_labels

                )

            )

        logging.info(

            "Train Samples : %d",

            len(train_results)

        )

        logging.info(

            "Validation Samples : %d",

            len(val_results)

        )

        return train_results, val_results

# ==========================================================
# Report Generator
# ==========================================================

class ReportGenerator:
    """
    数据预处理报告生成模块

    功能：
    1. 统计训练集和验证集信息
    2. 保存 JSON 报告
    3. 保存 Markdown 报告
    """

    def __init__(
        self,
        paths: ProjectPaths
    ):

        self.paths = paths

        self.paths.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    def _count_classes(
        self,
        image_infos: List[ImageInfo]
    ) -> Counter:

        counter = Counter()

        for image_info in image_infos:

            if image_info.primary_class is None:
                continue

            counter[
                image_info.primary_class
            ] += 1

        return counter


    def generate(
        self,
        train_infos: List[ImageInfo],
        val_infos: List[ImageInfo]
    ):

        train_counter = self._count_classes(
            train_infos
        )

        val_counter = self._count_classes(
            val_infos
        )


        report = {

            "dataset": {

                "train_images": len(train_infos),

                "val_images": len(val_infos),

                "total_images":
                    len(train_infos)
                    +
                    len(val_infos)

            },


            "class_distribution": {

                "train":
                    dict(train_counter),

                "val":
                    dict(val_counter)

            }

        }


        self._save_json(
            report
        )

        self._save_markdown(
            report
        )


        logging.info(
            "Report generated."
        )


    def _save_json(
        self,
        report: dict
    ):

        output = (
            self.paths.report_dir /
            "dataset_statistics.json"
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=4
            )


    def _save_markdown(
        self,
        report: dict
    ):

        output = (
            self.paths.report_dir /
            "preprocess_report.md"
        )


        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:


            f.write(
                "# Dataset Preprocess Report\n\n"
            )


            dataset = report["dataset"]


            f.write(
                "## Dataset Summary\n\n"
            )


            f.write(
                f"- Total Images: {dataset['total_images']}\n"
            )

            f.write(
                f"- Train Images: {dataset['train_images']}\n"
            )

            f.write(
                f"- Validation Images: {dataset['val_images']}\n\n"
            )


            f.write(
                "## Class Distribution\n\n"
            )


            for split in [
                "train",
                "val"
            ]:

                f.write(
                    f"### {split}\n\n"
                )

                for class_id, count in (
                    report["class_distribution"][split]
                    .items()
                ):

                    f.write(
                        f"- Class {class_id}: {count}\n"
                    )

                f.write("\n")

# ==========================================================
# Main Pipeline
# ==========================================================


def main():

    """
    YOLOv8 数据预处理主流程

    Pipeline:

    Scan
      ↓
    Clean
      ↓
    Process
      ↓
    Augment
      ↓
    Split
      ↓
    Report

    """


    logging.info(
        "=" * 60
    )

    logging.info(
        "Start preprocessing pipeline"
    )

    logging.info(
        "=" * 60
    )


    # ------------------------------------------------------
    # 初始化配置
    # ------------------------------------------------------

    config = PreprocessConfig()

    paths = ProjectPaths(
        config
    )

    if paths.processed_dir.exists():
        shutil.rmtree(paths.processed_dir)

    paths.processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    # ------------------------------------------------------
    # 1. Dataset Scan
    # ------------------------------------------------------

    scanner = DatasetScanner(
        config,
        paths
    )


    image_infos = scanner.scan()


    logging.info(
        "After scanning: %d images",
        len(image_infos)
    )


    # ------------------------------------------------------
    # 2. Image Cleaning
    # ------------------------------------------------------

    cleaner = ImageCleaner(
        config,
        paths
    )


    image_infos = cleaner.clean(
        image_infos
    )


    logging.info(
        "After cleaning: %d images",
        len(image_infos)
    )


    # ------------------------------------------------------
    # 3. Image Processing
    # ------------------------------------------------------

    processor = ImageProcessor(
        config,
        paths
    )


    image_infos = processor.process(
        image_infos
    )


    logging.info(
        "After processing: %d images",
        len(image_infos)
    )


    # ------------------------------------------------------
    # 4. Data Augmentation
    # ------------------------------------------------------

    augmentor = DataAugmentor(
        config,
        paths
    )


    image_infos = augmentor.augment(
        image_infos
    )


    logging.info(
        "After augmentation: %d images",
        len(image_infos)
    )


    # ------------------------------------------------------
    # 5. Dataset Split
    # ------------------------------------------------------

    splitter = DatasetSplitter(
        config,
        paths
    )


    train_infos, val_infos = splitter.split(
        image_infos
    )


    # ------------------------------------------------------
    # 6. Generate Report
    # ------------------------------------------------------

    reporter = ReportGenerator(
        paths
    )


    reporter.generate(
        train_infos,
        val_infos
    )
    return train_infos, val_infos

    logging.info(
        "=" * 60
    )

    logging.info(
        "Preprocessing completed successfully."
    )

    logging.info(
        "=" * 60
    )

# ==========================================================
# Program Entry
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        )

    )

    main()