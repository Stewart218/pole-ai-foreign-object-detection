"""
===========================================
EDA Analysis
Task3：探索性数据分析（EDA）
Author : Stewart218
===========================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # 设置Matplotlib中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print("Task3 - Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # -----------------------------
    # 读取Task2生成的数据
    # -----------------------------
    csv_path = r"E:\Pole_AI_Project\reports\image_information.csv"

    if not os.path.exists(csv_path):
        print(f"\n找不到文件：{csv_path}")
        print("请先运行 dataset_inspector.py")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    print("\n数据读取成功！")
    print(df.head())

    figure_dir = r"E:\Pole_AI_Project\results\figures"
    os.makedirs(figure_dir, exist_ok=True)

    # ==================================================
    # 一、类别统计
    # ==================================================
    print("\n" + "=" * 60)
    print("类别统计")
    print("=" * 60)

    class_count = df["Class"].value_counts().sort_index()

    # ==================================================
    # 分辨率统计
    # ==================================================

    resolution_count = df["Resolution"].value_counts().sort_index()

    print("\n" + "=" * 60)
    print("分辨率统计")
    print("=" * 60)

    print(resolution_count)

    # ==================================================
    # 分辨率分布图（Matplotlib）
    # ==================================================

    plt.figure(figsize=(8, 5))

    plt.bar(
        resolution_count.index,
        resolution_count.values
    )

    plt.title("Resolution Distribution")
    plt.xlabel("Resolution")
    plt.ylabel("Number of Images")

    # 在柱子上显示数量
    for i, value in enumerate(resolution_count.values):
        plt.text(
            i,
            value + 0.5,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figure_dir,
            "resolution_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    # ===============================
    # 类别数量柱状图（Matplotlib）
    # ===============================

    plt.figure(figsize=(8, 5))

    plt.bar(
        class_count.index,
        class_count.values
    )

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    # 在柱子上显示数字
    for i, value in enumerate(class_count.values):
        plt.text(
            i,
            value + 0.3,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figure_dir,
            "class_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    class_percent = (
            class_count / class_count.sum() * 100
    ).round(2)

    class_statistics = pd.DataFrame({
        "Count": class_count,
        "Percentage (%)": class_percent
    })

    print(class_statistics)

    # ==================================================
    # 二、图像宽度统计
    # ==================================================
    print("\n" + "=" * 60)
    print("图像宽度统计")
    print("=" * 60)

    width_statistics = df["Width"].describe()

    print(width_statistics)

    # ==================================================
    # 三、图像高度统计
    # ==================================================
    print("\n" + "=" * 60)
    print("图像高度统计")
    print("=" * 60)

    height_statistics = df["Height"].describe()

    print(height_statistics)

    # ==================================================
    # 四、整理统计结果
    # ==================================================

    resolution_statistics = pd.DataFrame({
        "Statistic": [
            "Width Max",
            "Width Min",
            "Width Mean",
            "Height Max",
            "Height Min",
            "Height Mean"
        ],
        "Value": [
            df["Width"].max(),
            df["Width"].min(),
            round(df["Width"].mean(), 2),

            df["Height"].max(),
            df["Height"].min(),
            round(df["Height"].mean(), 2)
        ]
    })

    print("\n")
    print(resolution_statistics)

    # ==================================================
    # 五、保存结果
    # ==================================================

    output_dir = r"E:\Pole_AI_Project\reports"

    class_statistics.to_csv(
        os.path.join(output_dir, "eda_class_statistics.csv"),
        encoding="utf-8-sig"
    )

    resolution_statistics.to_csv(
        os.path.join(output_dir, "eda_resolution_statistics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nEDA统计完成！")
    print("生成文件：")
    print("reports/eda_class_statistics.csv")
    print("reports/eda_resolution_statistics.csv")


if __name__ == "__main__":
    main()