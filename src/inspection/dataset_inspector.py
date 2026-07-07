"""
dataset_inspector.py

Task 2:
Inspect the raw dataset before annotation.

Current Functions:
1. Read dataset directory
2. Count images in each class
3. Calculate total number of images

Author: Stewart
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

# 当前脚本路径
CURRENT_FILE = Path(__file__).resolve()

# 项目根目录
PROJECT_ROOT = CURRENT_FILE.parents[2]

# 原始数据集目录
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "pole_foreign_objects"

# ==========================================================
# Supported Image Formats
# ==========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ==========================================================
# Check Dataset Directory
# ==========================================================

if not DATASET_DIR.exists():
    print("❌ Dataset folder does not exist.")
    print(f"Expected Path: {DATASET_DIR}")
    exit()

# ==========================================================
# Count Images
# ==========================================================

dataset_summary = {}

for class_folder in DATASET_DIR.iterdir():

    # 跳过非文件夹（例如 .DS_Store、json 文件）
    if not class_folder.is_dir():
        continue

    image_count = 0

    for file in class_folder.iterdir():

        if file.suffix.lower() in IMAGE_EXTENSIONS:
            image_count += 1

    dataset_summary[class_folder.name] = image_count

# ==========================================================
# Print Result
# ==========================================================

print("=" * 60)
print("Dataset Summary")
print("=" * 60)

total_images = 0

for class_name, image_count in dataset_summary.items():
    print(f"{class_name:<10} : {image_count:>5} images")
    total_images += image_count

print("-" * 60)
print(f"{'Total Images':<10} : {total_images:>5}")
print("=" * 60)