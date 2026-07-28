# Transmission Line Foreign Object Detection Dataset
# Final Dataset README


## 1. 数据集简介

本数据集用于输电线路异物智能检测任务，目标是利用深度学习目标检测模型自动识别输电线路附近的典型异物，包括：

- 鸟巢（bird_nest）
- 气球（balloon）
- 塑料袋（plastic_bag）
- 其他异物（other_foreign_object）


该数据集基于 YOLOv8 目标检测框架构建，用于训练和评估输电线路异物检测模型。


最终数据集用于项目最终模型训练：

- 模型：
  YOLOv8n

- 优化策略：
  类别均衡增强 + 图像增强 + 参数优化

- 最终模型：
  week3_aug_v2/best.pt


---

# 2. 数据集基本信息


| 参数 | 内容 |
|----|----|
| 数据集名称 | Transmission Line Foreign Object Dataset |
| 任务类型 | 目标检测(Object Detection) |
| 标注格式 | YOLO Detection Format |
| 图像格式 | PNG |
| 图像分辨率 | 1408 × 768 |
| 类别数量 | 4 |
| 检测目标 | 输电线路异物 |
| 标注工具 | LabelImg |
| 数据增强 | Brightness / HSV / Crop 等 |
| 检测框格式 | YOLO txt |



---

# 3. 数据类别说明


| ID | 类别名称 | 中文名称 |
|-|-|-|
|0|bird_nest|鸟巢|
|1|balloon|气球|
|2|plastic_bag|塑料袋|
|3|other_foreign_object|其他异物|



---

# 4. 原始数据集


## 4.1 数据来源


原始数据存放于：
data/yolo_dataset
结构：
yolo_dataset
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

---

## 4.2 原始数据规模


原始数据共：133 images

类别数量：

| 类别  |    数量 |
|-----|------:|
| 其他异物 |    30 |
| 塑料袋 |    33 |
| 气球  |    38 |
| 鸟巢  |    32 |


类别分布较均衡：

- 最大类别占比：
  气球 28.57%

- 最小类别占比：
  其他异物 22.56%


---

# 5. 数据处理流程


整体数据处理流程如下：
Raw Dataset
    |
    ↓
人工标注(LabelImg)
    |
    ↓
YOLO Dataset
    |
    ↓   
数据检查与清洗
    |
    ↓
数据增强
    |
    ↓
数据集划分
    |
    ↓
Final Dataset
    |
    ↓
YOLOv8训练
   

---

# 6. 数据清洗过程


## 6.1 图像质量检查


执行内容：

- 删除损坏图片
- 删除无法读取图片
- 检查图片格式
- 统一图像尺寸


检查结果：

- 图片均可正常读取
- 无损坏样本
- 分辨率统一为：1408×768

---

## 6.2 标签检查


检查内容：

- 标签文件是否存在
- 类别编号是否合法
- Bounding Box坐标是否符合YOLO格式


YOLO标签格式： class_id x_center y_center width height

所有标签均满足：0 <= x,y,w,h <=1

---

# 7. 数据增强策略


由于原始数据规模较小：133 images
直接训练容易出现：

- 泛化能力不足
- 小目标漏检
- 类别特征不足


因此进行了针对性数据增强。



---

# 7.1 亮度增强 Brightness


目的：

模拟不同天气、光照条件。


方法：

随机调整：

- 图像亮度
- 对比度


增强对象：

所有训练类别。



---

# 7.2 HSV颜色增强


模拟：

- 阴天
- 晴天
- 不同摄像头颜色偏差


调整：

- Hue
- Saturation
- Value



---

# 7.3 随机裁剪 Random Crop


目的：

增强小目标检测能力。


主要针对：

- 鸟巢
- 塑料袋
- 气球


处理：

裁剪后同步调整：

- bounding box坐标
- 标签位置



---

# 7.4 类别均衡增强


根据类别数量动态调整增强比例。


增强重点：

|类别|原因|
|-|-|
|塑料袋|检测困难，漏检较多|
|其他异物|样本复杂|
|气球|目标尺寸较小|


目标：

降低类别不均衡影响。


---

# 8. 数据集划分


最终数据集：final_dataset
结构：
final_dataset
    ├── images
    │   │
    │   ├── train
    │   ├── val
    │   └── test
    └── labels
        ├── train
        ├── val
        └── test

---

## 划分策略


采用：7 : 2 : 1

比例：

|集合|比例|用途|
|-|-|-|
|train|70%|模型训练|
|val|20%|模型调参|
|test|10%|最终独立测试|



注意：

训练集：包含增强样本

验证集和测试集：保持原始数据

避免增强数据导致评估指标虚高。


---

# 9. 最终样本分布


## Train


包含：

- 原始训练图片
- 增强图片


增强后：约数百张训练图片


主要增加：

- 亮度变化样本
- 色彩变化样本
- 小目标裁剪样本



---

## Validation


保持：原始图片

用于：

- 超参数调整
- 模型选择



---

## Test


保持：完全独立原始图片

用于最终评价。


---

# 10. 优化亮点


## 10.1 面向输电线路场景优化


不同于通用目标检测数据集：

输电线路异物具有：

- 目标尺寸小
- 背景复杂
- 类别差异小


因此增强策略针对实际场景设计。


---

## 10.2 小目标检测增强


通过：

- 随机裁剪
- 尺寸变化
- 类别增强


提升模型对：

- 小鸟巢
- 塑料袋
- 气球

检测能力。



# 11. 模型训练结果


最终模型：YOLOv8n + Class Balanced Augmentation

测试集结果：


|指标|结果|
|-|-:|
|Precision|0.900|
|Recall|0.917|
|mAP50|0.913|
|mAP50-95|0.116

---

# 12. 数据集使用方法


训练：

```bash
python train_aug_v2.py

配置：
data/final_dataset/dataset.yaml

验证：
python evaluate_test.py














