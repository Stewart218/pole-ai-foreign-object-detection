# 模型选型与基线方案

## 1. 项目背景

本项目针对输电线路场景下的异物智能检测任务，目标是利用深度学习目标检测算法，实现对输电杆塔及线路区域中潜在危险异物的自动识别。

主要检测类别包括：

| 类别 | 英文名称 |
|---|---|
| 鸟巢 | bird_nest |
| 气球 | balloon |
| 塑料袋 | plastic_bag |
| 其他异物 | other_foreign_object |

传统人工巡检方式存在效率低、周期长、受环境影响明显等问题，因此引入计算机视觉技术，实现输电线路异物自动检测，提高巡检智能化水平。


---

# 2. 数据集初步分析

对原始数据集进行统计分析。

## 2.1 数据规模

原始数据集共包含：

- 图片数量：133张
- 图像格式：PNG
- 图像尺寸：1408 × 768

## 2.2 类别分布


|类别|数量|比例|
|-|-|-|
|鸟巢|32|24.06%|
|气球|38|28.57%|
|塑料袋|33|24.81%|
|其他异物|30|22.56%|


数据整体类别较均衡，没有明显类别缺失问题。

但存在以下特点：

1. 数据规模较小；
2. 场景较单一；
3. 异物目标尺寸较小；
4. 部分异物与背景颜色接近。


因此模型需要具备：

- 较强的小目标检测能力；
- 较好的泛化能力；
- 较低计算资源需求。


---


# 3. YOLOv8模型选型


## 3.1 YOLOv8简介


YOLOv8 是 Ultralytics 公司推出的新一代目标检测模型，相比 YOLOv5、YOLOv7，在网络结构、训练策略以及检测精度方面进行了优化。


YOLOv8主要由以下部分组成：

Input Image 
   ↓
Backbone
(CSPDarknet)
   ↓
Neck
(FPN + PAN)
   ↓
Detection Head
  ↓
Bounding Box + Class Prediction

---

# 4. YOLOv8算法原理简述


## 4.1 Backbone特征提取


YOLOv8使用改进后的CSP结构进行特征提取。

主要作用：

- 提取目标纹理信息；
- 学习不同尺度目标特征；
- 降低计算量。


---

## 4.2 Neck多尺度特征融合


YOLOv8采用：

- FPN
- PAN


结构融合不同尺度特征。


作用：

对于输电线路异物检测：

- 鸟巢尺寸较大；
- 气球、塑料袋尺寸较小；

多尺度融合能够提高小目标检测能力。


---

## 4.3 Detection Head


YOLOv8采用：

- Anchor-Free检测方式；
- Decoupled Head结构。


分别预测：

1. 分类概率：

[P(class)]


2. 边界框位置：

[(x,y,w,h)]


最终输出：

类别
置信度
目标框坐标

---

# 6. YOLOv8模型规模选择


Ultralytics YOLOv8提供多个模型规模：


|模型|参数量|速度|精度|
|-|-|-|-|
|YOLOv8n|最低|最快|较低|
|YOLOv8s|中等|较快|较高|
|YOLOv8m|较高|一般|更高|


本项目选择：

# YOLOv8n


原因：

## 1. 数据量较小

当前数据集： 133 images
若使用大型模型：

容易产生：

- 过拟合；
- 泛化下降。


---

## 2. 满足实时检测需求


输电线路巡检通常要求：

- 快速推理；
- 较低计算资源。


YOLOv8n具有：

- 参数少；
- FPS高；
- 部署方便。


---

## 3. 作为Baseline便于后续优化


Baseline模型用于建立初始性能基准。


后续可通过：

- 数据增强；
- 优化器调整；
- 超参数调整；

进一步提升性能。


---

# 7. 基线训练方案


## 模型
YOLOv8n

## 输入尺寸
640 × 640

## 数据划分


采用：
Train : Val : Test
7 : 2 : 1

## 训练参数


|参数|设置|
|-|-|
|Epoch|100|
|Batch Size|16|
|Optimizer|AdamW|
|Image Size|640|
|Learning Rate|0.001|
|Device|CUDA GPU|


---

# 8. 最小可运行推理代码


## 环境要求


```bash
pip install ultralytics
```

## predict.py

from ultralytics import YOLO


def main():

    # 加载训练好的模型
    model = YOLO(
        "runs/train/baseline_yolov8n/weights/best.pt"
    )


    # 输入图片
    image_path = (
        "test.jpg"
    )


    # 推理
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25
    )


    # 显示检测结果
    for result in results:

        result.show()

        # 保存结果
        result.save(
            filename="prediction.jpg"
        )


if __name__ == "__main__":
    main()


运行后：
输入：

test.jpg

输出：

prediction.jpg

结果包含：
检测框；
类别名称；
置信度。