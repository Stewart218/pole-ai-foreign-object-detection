"""
============================================================
功能：
    单独评估模型
============================================================
"""
from pathlib import Path
from ultralytics import YOLO


# ============================
# 项目根目录
# ============================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================
# 模型路径
# ============================

MODEL_PATH = (
    PROJECT_ROOT
    /
    "runs"
    /
    "train"
    /
    "baseline"
    /
    "weights"
    /
    "best.pt"
)


# ============================
# 最终数据集
# ============================

DATASET_YAML = (
    PROJECT_ROOT
    /
    "data"
    /
    "final_dataset"
    /
    "dataset.yaml"
)



def main():

    print("=" * 60)
    print("Final Model Evaluation")
    print("=" * 60)


    print(f"Project root:\n{PROJECT_ROOT}")

    print(f"\nModel:")
    print(MODEL_PATH)

    print(f"\nDataset:")
    print(DATASET_YAML)



    # 检查路径

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found:\n{DATASET_YAML}"
        )


    # 加载模型

    model = YOLO(
        str(MODEL_PATH)
    )


    # ============================
    # test集评估
    # ============================

    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        device=0
    )


    # ============================
    # 输出指标
    # ============================

    print("\n" + "=" * 60)
    print("Evaluation Result")
    print("=" * 60)


    print(
        f"Precision: {results.box.mp:.4f}"
    )

    print(
        f"Recall: {results.box.mr:.4f}"
    )

    print(
        f"mAP50: {results.box.map50:.4f}"
    )

    print(
        f"mAP50-95: {results.box.map:.4f}"
    )



if __name__ == "__main__":
    main()