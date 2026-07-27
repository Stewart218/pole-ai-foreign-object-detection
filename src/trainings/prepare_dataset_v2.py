"""
============================================================
prepare_dataset_v2.py

Week3 Dataset Preparation

功能:
1. 读取增强后的YOLO数据集
2. train/val/test划分
3. 图片标签同步复制
4. 自动生成dataset.yaml

输入:

data/
└── augmented_dataset
    ├── images
    │   └── train
    └── labels
        └── train


输出:

data/
└── processed_augmented
    |
    ├── images
    │   ├── train
    │   ├── val
    │   └── test
    |
    ├── labels
    │   ├── train
    │   ├── val
    │   └── test
    |
    ├── dataset.yaml
    |
    └── split_report.txt


============================================================
"""


from pathlib import Path
import random
import shutil
import yaml



# ============================================================
# Path Configuration
# ============================================================


class DatasetConfig:


    # 输入增强数据

    source_dir = Path(
        r"E:\Pole_AI_Project\data\augmented_dataset"
    )


    image_dir = (
        source_dir
        /
        "images"
        /
        "train"
    )


    label_dir = (
        source_dir
        /
        "labels"
        /
        "train"
    )



    # 输出目录

    output_dir = Path(
        r"E:\Pole_AI_Project\data\processed_augmented"
    )


    image_output = (
        output_dir
        /
        "images"
    )


    label_output = (
        output_dir
        /
        "labels"
    )



    yaml_path = (
        output_dir
        /
        "dataset.yaml"
    )



    report_path = (
        output_dir
        /
        "split_report.txt"
    )



    # 划分比例

    train_ratio = 0.7

    val_ratio = 0.2

    test_ratio = 0.1



    seed = 42



    classes = [

        "bird_nest",

        "balloon",

        "plastic_bag",

        "other_foreign_object"

    ]





# ============================================================
# Create Directory
# ============================================================


def create_dirs():

    cfg = DatasetConfig()


    for split in [

        "train",

        "val",

        "test"

    ]:


        (
            cfg.image_output
            /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


        (
            cfg.label_output
            /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )




# ============================================================
# Dataset Split
# ============================================================


def split_dataset():

    cfg = DatasetConfig()


    images = list(

        cfg.image_dir.glob(
            "*.png"
        )

    )


    print("="*60)

    print(
        "Prepare Augmented Dataset"
    )

    print(
        f"Total images: {len(images)}"
    )

    print("="*60)



    random.seed(
        cfg.seed
    )


    random.shuffle(
        images
    )



    total=len(images)


    train_num=int(
        total *
        cfg.train_ratio
    )


    val_num=int(
        total *
        cfg.val_ratio
    )



    train_images = images[:train_num]


    val_images = images[
        train_num:
        train_num+val_num
    ]


    test_images = images[
        train_num+val_num:
    ]



    dataset={

        "train":
        train_images,

        "val":
        val_images,

        "test":
        test_images

    }



    for split,files in dataset.items():


        print(
            f"{split}: {len(files)}"
        )


        for img_path in files:



            label_path=(

                cfg.label_dir

                /

                (
                    img_path.stem
                    +
                    ".txt"
                )

            )


            if not label_path.exists():

                print(
                    "Missing label:",
                    img_path.name
                )

                continue



            shutil.copy(

                img_path,

                cfg.image_output
                /
                split
                /
                img_path.name

            )



            shutil.copy(

                label_path,

                cfg.label_output
                /
                split
                /
                label_path.name

            )



    return dataset




# ============================================================
# Generate dataset.yaml
# ============================================================


def generate_yaml():

    cfg=DatasetConfig()


    data={


        "path":

        str(
            cfg.output_dir
        ),



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

            in enumerate(
                cfg.classes
            )

        }

    }



    with open(

        cfg.yaml_path,

        "w",

        encoding="utf-8"

    ) as f:


        yaml.dump(

            data,

            f,

            allow_unicode=True,

            sort_keys=False

        )



    print(
        "dataset.yaml generated:"
    )


    print(
        cfg.yaml_path
    )





# ============================================================
# Generate Report
# ============================================================


def generate_report(dataset):


    cfg=DatasetConfig()


    with open(

        cfg.report_path,

        "w",

        encoding="utf-8"

    ) as f:


        f.write(

            "Dataset Split Report\n"

        )


        f.write(

            "="*40+"\n\n"

        )



        for k,v in dataset.items():


            f.write(

                f"{k}: {len(v)} images\n"

            )



        f.write(

            "\nRatio:\n"

        )


        f.write(

            "train=0.7\n"

        )

        f.write(

            "val=0.2\n"

        )

        f.write(

            "test=0.1\n"

        )



    print(

        "Report saved:",

        cfg.report_path

    )





# ============================================================
# Main
# ============================================================


def main():


    create_dirs()


    dataset = split_dataset()


    generate_yaml()


    generate_report(
        dataset
    )


    print(
        "\nDataset preparation finished!"
    )




if __name__=="__main__":

    main()