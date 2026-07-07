from pathlib import Path

# 当前文件路径
CURRENT_FILE = Path(__file__).resolve()

# 项目根目录
PROJECT_ROOT = CURRENT_FILE.parents[2]

# 数据集路径
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "pole_foreign_objects"

# 支持的图片格式
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# 检查数据集是否存在
if not DATASET_DIR.exists():
    print("❌ Dataset folder does not exist.")
    exit()

print("=" * 50)
print("Dataset Categories")
print("=" * 50)

# 遍历每个类别文件夹
for class_folder in DATASET_DIR.iterdir():

    # 只处理文件夹
    if class_folder.is_dir():

        # 当前类别图片数量
        image_count = 0

        # 遍历类别中的所有文件
        for file in class_folder.iterdir():

            # 判断是否是图片
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1

        print(f"{class_folder.name}: {image_count} images")