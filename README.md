# 输电线路异物智能检测系统
## Pole AI - Foreign Object Detection Based on YOLOv8


<div align="center">

基于深度学习目标检测技术的输电线路异物自动识别系统

YOLOv8 | Computer Vision | Power Grid Inspection

</div>


---

# 1. 项目简介

## 1.1 项目背景

输电线路长期运行过程中，可能受到鸟巢、塑料袋、气球以及其他异物影响，导致线路短路、跳闸甚至电网事故。

传统人工巡检方式存在：

- 检测周期长；
- 人力成本高；
- 特殊环境下巡检困难；
- 主观因素影响较大。

本项目利用计算机视觉与深度学习目标检测技术，实现输电线路场景下异物自动检测，提高输电线路智能巡检水平。


---

## 1.2 项目目标

构建一个完整的输电线路异物检测系统，实现：

- 异物目标自动定位；
- 异物类别自动识别；
- 检测结果可视化展示；
- 模型训练与推理流程可复现。


检测类别：

|类别|英文类别|
|-|-|
|鸟巢|bird_nest|
|气球|balloon|
|塑料袋|plastic_bag|
|其他异物|other_foreign_object|


---

# 2. 技术路线


整体流程：

原始巡检图片

    ↓
数据清洗与标注

    ↓
数据增强与类别优化

    ↓
YOLOv8模型训练

    ↓
模型优化实验 

    ↓
独立测试集评估 

    ↓
用户交互Demo展示

---

# 3. 项目特点


## 3.1 数据优化

针对输电线路异物检测特点：

- 小目标占比较高；
- 样本数量有限；
- 类别存在差异；

进行了：

- 图像清洗；
- 数据增强；
- HSV颜色增强；
- 亮度增强；
- 类别均衡优化。


---

## 3.2 模型优化


完成多轮实验：

|实验|模型|
|-|-|
|Experiment 1|YOLOv8n Baseline|
|Experiment 2|YOLOv8n + AdamW优化|
|Experiment 3|YOLOv8s|
|Experiment 4|YOLOv8n + Class Balanced Augmentation|


最终选择综合性能最佳模型。


---

# 4. 环境配置


## 4.1 硬件环境


|配置|参数|
|-|-|
|GPU|NVIDIA RTX 4060 Laptop GPU|
|CUDA|12.6|
|显存|8GB|


---

## 4.2 软件环境


|软件|版本|
|-|-|
|Python|3.9|
|PyTorch|2.8.0|
|Ultralytics|8.4.89|
|CUDA Toolkit|12.6|


---

# 5. 环境安装


## 5.1 创建conda环境


```bash
conda create -n pole_ai39 python=3.9
```

进入环境：
```bash
conda activate pole_ai39
```

## 5.2 安装依赖


```bash
pip install -r requirements.txt
```

主要依赖：

ultralytics
torch
torchvision
opencv-python
numpy
pandas
matplotlib

# 6. 项目目录结构


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


# 7. 数据集说明


最终数据集路径：data/final_dataset

数据划分：

Train : Validation : Test

7 : 2 : 1

包含：

images/
labels/
dataset.yaml

详细说明：

data/final_dataset/final_dataset_readme.md

# 8. 模型训练


## 8.1 Baseline训练

运行：
```bash
python src/trainings/train_experiments/train_baseline.py
```

## 8.2 数据增强优化训练

运行：
```bash
python src/trainings/train_experiments/train_aug_v2.py
```

训练完成后生成：

runs/train/
    weights/
        best.pt

最终模型复制：
models/best_model.pt


# 9. 模型评估

使用独立测试集：

```bash
python src/trainings/evaluate_test.py
```

评估指标：
    Precision
    Recall
    mAP@0.5
    mAP@0.5:0.95
    推理速度

# 10. Demo使用方法


## 10.1 功能介绍

Demo支持：
    用户自主选择图片；
    自动加载最佳模型；
    自动检测异物；
    绘制检测框；
    显示类别与置信度；
    保存检测结果。

## 10.2 启动Demo

运行：

```bash
python src/inference/demo.py
```

## 10.3 操作流程

启动程序

    ↓

点击选择图片

    ↓

加载best_model.pt

    ↓

YOLOv8推理

    ↓

显示检测结果

    ↓

保存输出图片

输出路径：

demo_results/


# 11. 核心实验结果


## 11.1 模型性能对比

| 模型                    | Precision | Recall | mAP50 |
|-----------------------|-----------|--------|-------|
| YOLOv8n Baseline      | 0.649     | 0.667  | 0.657 |
| YOLOv8n + AdamW       | 0.703     | 0.658  | 0.655 |
| YOLOv8s               | 0.287     | 0.449  | 0.342 |
| YOLOv8n + Augmentation | 0.900     | 0.917  | 0.913 |

最终模型：

YOLOv8n + Class Balanced Augmentation


## 11.2 最终模型优势

相比Baseline：
    Precision提升；
    Recall提升；
    小目标检测能力增强；
    漏检情况减少；
    泛化能力提高。


# 13. 项目总结

本项目完成了从：

数据构建

    ↓

模型选型

    ↓

Baseline建立

    ↓

模型优化

    ↓

性能评估

    ↓

Demo部署

的完整深度学习目标检测流程。
最终实现输电线路异物智能检测系统，为电网智能巡检提供技术方案。