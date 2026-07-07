from pathlib import Path

# 当前文件(dataset_inspector.py)的位置
CURRENT_FILE = Path(__file__).resolve()

# 项目根目录（向上两级）
PROJECT_ROOT = CURRENT_FILE.parents[2]

# 数据集路径
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "pole_foreign_objects"

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
        print(class_folder.name)