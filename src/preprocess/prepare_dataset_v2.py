"""
============================================================
Prepare Final Dataset V2

功能:
1. 构建最终数据集 final_dataset
2. train:
      使用增强后的训练数据
3. val:
      使用原始验证集
4. test:
      使用独立测试集
5. 自动生成 dataset.yaml
6. 输出数据统计报告

============================================================
"""


from pathlib import Path
import shutil
import yaml
import logging


# ============================================================
# 项目路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DATA_DIR = PROJECT_ROOT / "data"



# ============================================================
# 输入数据
# ============================================================


# 增强后的训练数据

AUGMENTED_DATASET = (
    DATA_DIR /
    "augmented_dataset"
)



# 原始划分数据

PROCESSED_SPLIT = (
    DATA_DIR /
    "processed" /
    "split"
)



# ============================================================
# 输出最终数据集
# ============================================================


FINAL_DATASET = (
    DATA_DIR /
    "final_dataset_v2"
)



# ============================================================
# 类别
# ============================================================


CLASS_NAMES = [

    "bird_nest",

    "balloon",

    "plastic_bag",

    "other_foreign_object"

]



# ============================================================
# 日志
# ============================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)



# ============================================================
# 创建目录
# ============================================================


def create_directories():

    dirs = [

        FINAL_DATASET /
        "images/train",

        FINAL_DATASET /
        "images/val",

        FINAL_DATASET /
        "images/test",


        FINAL_DATASET /
        "labels/train",

        FINAL_DATASET /
        "labels/val",

        FINAL_DATASET /
        "labels/test",

    ]


    for d in dirs:

        d.mkdir(
            parents=True,
            exist_ok=True
        )



# ============================================================
# 复制图片和标签
# ============================================================


def copy_split_dataset(
        src_images,
        src_labels,
        dst_images,
        dst_labels
):

    """
    复制YOLO格式数据

    image:
        xxx.jpg/png

    label:
        xxx.txt

    """

    count = 0


    if not src_images.exists():

        raise FileNotFoundError(
            f"Image folder not found:\n{src_images}"
        )


    for img in src_images.iterdir():


        if img.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            continue



        label = (
            src_labels /
            f"{img.stem}.txt"
        )


        shutil.copy2(
            img,
            dst_images /
            img.name
        )


        if label.exists():

            shutil.copy2(
                label,
                dst_labels /
                label.name
            )


        count += 1


    return count



# ============================================================
# 构建final_dataset
# ============================================================


def build_final_dataset():


    logging.info("=" * 60)

    logging.info(
        "Building Final Dataset"
    )

    logging.info("=" * 60)



    create_directories()



    # --------------------------------------------------------
    # train
    # --------------------------------------------------------


    logging.info(
        "\n[1/3] Copy augmented train dataset..."
    )


    train_num = copy_split_dataset(

        AUGMENTED_DATASET /
        "images/train",

        AUGMENTED_DATASET /
        "labels/train",


        FINAL_DATASET /
        "images/train",

        FINAL_DATASET /
        "labels/train"

    )



    logging.info(
        f"Train images: {train_num}"
    )



    # --------------------------------------------------------
    # val
    # --------------------------------------------------------


    logging.info(
        "\n[2/3] Copy original validation dataset..."
    )


    val_num = copy_split_dataset(

        PROCESSED_SPLIT /
        "images/val",

        PROCESSED_SPLIT /
        "labels/val",


        FINAL_DATASET /
        "images/val",

        FINAL_DATASET /
        "labels/val"

    )



    logging.info(
        f"Validation images: {val_num}"
    )



    # --------------------------------------------------------
    # test
    # --------------------------------------------------------


    logging.info(
        "\n[3/3] Copy original test dataset..."
    )


    test_num = copy_split_dataset(

        PROCESSED_SPLIT /
        "images/test",

        PROCESSED_SPLIT /
        "labels/test",


        FINAL_DATASET /
        "images/test",

        FINAL_DATASET /
        "labels/test"

    )



    logging.info(
        f"Test images: {test_num}"
    )


    return {

        "train": train_num,

        "val": val_num,

        "test": test_num

    }



# ============================================================
# 生成dataset.yaml
# ============================================================


def generate_dataset_yaml():


    yaml_path = (
        FINAL_DATASET /
        "dataset.yaml"
    )


    config = {


        "path":
        str(FINAL_DATASET)
        .replace("\\", "/"),


        "train":
        "images/train",


        "val":
        "images/val",


        "test":
        "images/test",



        "names":
        {
            i:name
            for i,name
            in enumerate(CLASS_NAMES)
        }

    }



    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:


        yaml.dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False
        )



    logging.info(
        "\nDataset yaml generated:"
    )

    logging.info(
        yaml_path
    )



# ============================================================
# 数据统计
# ============================================================


def generate_report(stats):


    report = (

        "============================\n"
        "Final Dataset Report\n"
        "============================\n\n"

        f"Train images: {stats['train']}\n"

        f"Val images:   {stats['val']}\n"

        f"Test images:  {stats['test']}\n\n"


        "Dataset strategy:\n"

        "- Train: augmented dataset\n"

        "- Val: original dataset\n"

        "- Test: independent original dataset\n"

    )



    report_path = (
        FINAL_DATASET /
        "dataset_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)



    logging.info(
        "\nReport saved:"
    )

    logging.info(
        report_path
    )



# ============================================================
# Main
# ============================================================


def main():


    stats = build_final_dataset()


    generate_dataset_yaml()


    generate_report(stats)



    logging.info(
        "\n"
        "="*60
    )

    logging.info(
        "Final dataset preparation completed!"
    )

    logging.info(
        "="*60
    )



if __name__ == "__main__":

    main()