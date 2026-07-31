# Pole AI Foreign Object Detection
# Project Structure


## 1. Project Overview

本项目基于 YOLOv8 深度学习目标检测算法，实现输电线路异物智能检测。

主要功能：

- 输电线路异物数据处理
- YOLO格式数据集构建
- 数据增强与类别均衡优化
- 多模型训练实验
- 模型性能评估
- 最优模型推理部署


检测目标：

|Class ID|Category|
|-|-|
|0|bird_nest|
|1|balloon|
|2|plastic_bag|
|3|other_foreign_object|


最终模型：YOLOv8n + Class Balanced Augmentation
权重：models/best_model.pt


---

# 2. Directory Structure

Pole_AI_Project

    │
    ├── README.md
    ├── PROJECT_STRUCTURE.md
    ├── requirements.txt
    ├── demo.py
    │
    ├── data
    │   │
    │   ├── raw
    │   │   └── pole_foreign_objects
    │   │       ├── 鸟巢
    │   │       ├── 气球
    │   │       ├── 塑料袋
    │   │       └── 其他异物
    │   │
    │   │
    │   ├── annotation
    │   │   ├── images
    │   │   │   ├── train
    │   │   │   └── val
    │   │   │
    │   │   └── labels
    │   │       ├── train
    │   │       └── val
    │   │
    │   │
    │   ├── augmented_dataset
    │   │   ├── images
    │   │   └── labels
    │   │
    │   │
    │   ├── processed
    │   │   ├── split
    │   │   │   ├── train
    │   │   │   ├── val
    │   │   │   └── test
    │   │   │
    │   │   └── dataset.yaml
    │   │
    │   │
    │   └── final_dataset
    │       │
    │       ├── images
    │       │   ├── train
    │       │   ├── val
    │       │   └── test
    │       │
    │       ├── labels
    │       │   ├── train
    │       │   ├── val
    │       │   └── test
    │       │
    │       ├── dataset.yaml
    │       └── final_dataset_readme.md
    │
    │
    ├── models
    │   │
    │   └── best_model.pt
    │
    │
    ├── src
    │   │
    │   ├──preprocess
    │   │   │
    │   │   ├── augment_yolo_dataset_v2.py
    │   │   ├── prepare_dataset_v2.py
    │   │   └── preprocess.py
    │   │
    │   ├── trainings
    │   │   │
    │   │   ├── compare_prediction.py
    │   │   ├── predict_final.py
    │   │   ├── visualize_results.py
    │   │   │
    │   │   └── train_experiments
    │   │       ├── train_aug_v2.py
    │   │       ├── train_baseline.py
    │   │       ├── train_lr_opt.py
    │   │       └── train_yolov8s.py
    │   │   
    │   ├── visualize_results.py
    │   │
    │   └── evaluation
    │      │
    │      ├── evaluate_test.py
    │      └── evaluate_all_models.py
    │   
    │
    ├── runs
    │   │
    │   └── train
    │       │
    │       ├── baseline_yolov8n
    │       ├── week3_aug_v2
    │       ├── week3_yolov8s
    │       └── week3_lr_opt
    │
    │
    ├── docs
    │    │
    │    ├── model_comparison.md
    │    ├── model_technical_summary.md
    │    ├── eveluation_metrics.md
    │    └── final_model_evaluation_report.md
    │
    ├── results
    │   │
    │   ├── week2
    │   │
    │   └── week3
    │
    │
    └──demo_results
---

# 3. Directory Description


## data/

用于保存所有数据相关文件。


### raw/

原始采集数据：

- 未标注
- 未增强
- 原始图片


### annotation/

人工标注后的初始YOLO数据集。


包含：
images/
labels/

---

### augmented_dataset/

保存数据增强产生的图片。


增强方式：

- Brightness
- HSV
- Random Crop


---

### processed/

中间处理数据。


包括：

- 清洗后的数据
- 数据划分
- dataset.yaml


---

### final_dataset/

最终用于模型训练和测试的数据集。


特点：

- 训练集包含增强数据
- 验证集保持原始数据
- 测试集保持独立


---

# 4. src/说明


## preprocess/


负责：

- 数据检查
- 数据增强
- 数据集生成



主要脚本：

|文件|功能|
|-|-|
|augment_yolo_dataset_v2.py|增强生成|
|prepare_dataset_v2.py|最终数据划分|



---

## trainings/


负责模型训练。


包含：

- baseline实验
- 参数优化实验
- 模型结构实验
- 数据增强实验


---

## evaluation/


负责模型统一评价。


输出：

- Precision
- Recall
- mAP50
- mAP50-95
- 推理速度


---

## inference/


负责：

- 图片预测
- 困难样本测试
- 结果可视化

## demo_results/

保存demo.py检测结果。

## runs/train/

yolo训练输出

各训练模型权重、各指标图





