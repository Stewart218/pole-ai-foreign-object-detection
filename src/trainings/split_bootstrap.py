import random
import shutil
from pathlib import Path

# ===========================
# Configuration
# ===========================
PROJECT_ROOT = Path(r"E:\Pole_AI_Project")

SOURCE_IMAGES = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "train"
SOURCE_LABELS = PROJECT_ROOT / "data" / "yolo_dataset" / "labels" / "train"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "bootstrap_dataset"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ===========================
# Remove old bootstrap dataset
# ===========================
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

# Create folders
for folder in [
    OUTPUT_ROOT / "images" / "train",
    OUTPUT_ROOT / "images" / "val",
    OUTPUT_ROOT / "labels" / "train",
    OUTPUT_ROOT / "labels" / "val",
]:
    folder.mkdir(parents=True, exist_ok=True)

# ===========================
# Find labeled images
# ===========================
samples = []

for label_file in SOURCE_LABELS.glob("*.txt"):
    image_file = SOURCE_IMAGES / (label_file.stem + ".png")

    if image_file.exists():
        samples.append((image_file, label_file))

print(f"Found {len(samples)} labeled images.")

# ===========================
# Split
# ===========================
random.seed(RANDOM_SEED)
random.shuffle(samples)

split_index = int(len(samples) * TRAIN_RATIO)

train_samples = samples[:split_index]
val_samples = samples[split_index:]

# ===========================
# Copy files
# ===========================
def copy_dataset(dataset, image_dir, label_dir):
    for image_file, label_file in dataset:
        shutil.copy2(image_file, image_dir / image_file.name)
        shutil.copy2(label_file, label_dir / label_file.name)


copy_dataset(
    train_samples,
    OUTPUT_ROOT / "images" / "train",
    OUTPUT_ROOT / "labels" / "train",
)

copy_dataset(
    val_samples,
    OUTPUT_ROOT / "images" / "val",
    OUTPUT_ROOT / "labels" / "val",
)

print(f"Train: {len(train_samples)}")
print(f"Val:   {len(val_samples)}")
print("Bootstrap dataset created successfully.")