import shutil
from pathlib import Path

# ==========================
# 路径设置
# ==========================
labels_dir = Path(r"E:\Pole_AI_Project\data\yolo_dataset\labels\train")

image_folders = [
    Path(r"E:\Pole_AI_Project\data\yolo_dataset\images\train\其他异物"),
    Path(r"E:\Pole_AI_Project\data\yolo_dataset\images\train\塑料袋"),
    Path(r"E:\Pole_AI_Project\data\yolo_dataset\images\train\气球"),
    Path(r"E:\Pole_AI_Project\data\yolo_dataset\images\train\鸟巢"),
]

target_dir = Path(r"E:\Pole_AI_Project\data\yolo_dataset\images\train")
target_dir.mkdir(parents=True, exist_ok=True)

# 图片可能的格式
image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

# ==========================
# 开始复制
# ==========================
copied = 0
skipped = 0
not_found = []

label_files = list(labels_dir.glob("*.txt"))

print(f"共找到 {len(label_files)} 个标签文件。\n")

for label_file in label_files:
    image_name = label_file.stem
    found = False

    for folder in image_folders:
        for ext in image_extensions:
            img_path = folder / f"{image_name}{ext}"

            if img_path.exists():
                dst = target_dir / img_path.name

                if dst.exists():
                    print(f"跳过（已存在）：{img_path.name}")
                    skipped += 1
                else:
                    shutil.copy2(img_path, dst)
                    print(f"复制成功：{img_path.name}")
                    copied += 1

                found = True
                break

        if found:
            break

    if not found:
        not_found.append(image_name)

# ==========================
# 输出结果
# ==========================
print("\n========== 完成 ==========")
print(f"复制成功：{copied}")
print(f"已存在跳过：{skipped}")
print(f"未找到图片：{len(not_found)}")

if not_found:
    print("\n以下标签未找到对应图片：")
    for name in not_found:
        print(name)