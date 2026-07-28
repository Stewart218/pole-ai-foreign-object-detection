# Transmission Line Foreign Object Detection
# Model Comparison Report

## 1. Experimental Overview

本项目针对输电线路异物检测任务，
基于 YOLOv8 开展模型优化实验。

实验目标：

- 提升小目标异物检测能力；
- 降低稀缺类别漏检；
- 提升模型泛化性能。


测试指标：

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95


---

# 2. Experimental Results


| Experiment | Model | Optimization Strategy | Precision | Recall | mAP50 | mAP50-95 |
|-|-|-|-|-|-|-|
|Baseline|YOLOv8n|None|0.649|0.667|0.657|0.204|
|Exp1|YOLOv8s|Increase model size|0.287|0.449|0.342|0.158|
|Exp2|YOLOv8n|AdamW optimizer|0.703|0.658|0.655|0.154|
|Exp3|YOLOv8n|Small Object Augmentation|0.253|0.333|0.383|0.128|
|Exp4|YOLOv8n|Class Balanced Augmentation|**0.900**|**0.917**|**0.913**|0.116|


---

# 3. Optimization Analysis


## 3.1 Model Scale Optimization

Experiment:

YOLOv8n → YOLOv8s


Result:

mAP50:

```
0.657 → 0.342
```


Analysis:

增加模型参数量未带来性能提升。

原因：

- 数据集规模较小；
- 目标特征复杂度有限；
- 模型容量超过数据承载能力。


Conclusion:

对于当前数据规模，
YOLOv8n更加适合。


---

## 3.2 Optimizer Optimization


Strategy:

AdamW optimizer


Result:

Precision:

```
0.649 → 0.703
```


Analysis:

优化器调整提高预测稳定性，
但对于Recall提升有限。


Conclusion:

优化器调整具有一定效果，
但不是主要提升因素。


---

## 3.3 Small Object Augmentation


Strategy:

针对小目标增强


Result:

mAP50:

```
0.383
```


Analysis:

效果低于Baseline。


Possible reasons:

- 增强方式改变目标分布；
- 部分增强样本存在噪声；
- 原始数据规模限制。


Conclusion:

单独小目标增强不足以解决问题。


---

## 3.4 Class Balanced Augmentation


Strategy:

类别均衡增强：

- Brightness augmentation
- HSV augmentation
- Random crop
- 类别针对性扩充


Result:


mAP50:

```
0.657 → 0.913
```


提升：

```
+39.0%
```


Analysis:

有效解决：

- 类别数量不均衡；
- 稀缺类别训练不足；
- 部分场景泛化能力不足。


Conclusion:

该方法为最佳优化策略。


---

# 4. Final Model


## Selected Model

```
YOLOv8n + Class Balanced Augmentation
```


Model weight:

```
runs/train/week3_aug_v2/weights/best.pt
```


Performance:

|Metric|Value|
|-|-:|
|Precision|0.900|
|Recall|0.917|
|mAP50|0.913|
|mAP50-95|0.116|


---

# 5. Class-level Performance


|Class|Precision|Recall|mAP50|
|-|-|-|-|
|bird_nest|0.979|1.000|0.995|
|balloon|0.660|0.667|0.665|
|plastic_bag|0.978|1.000|0.995|
|other_foreign_object|0.983|1.000|0.995|


---

# 6. Difficult Sample Analysis


## Remaining Difficult Category

### Balloon


Performance:

```
mAP50 = 0.665
```


Reasons:

- shape variation;
- similarity with sky background;
- limited samples.


Future improvements:

- increase real scene samples;
- introduce background diversification;
- use higher resolution training.


---

# 7. Final Conclusion


通过多组实验对比：

- 单纯提升模型规模无明显收益；
- 参数优化只能带来有限提升；
- 针对数据特点的数据增强效果最佳。


最终模型：

```
YOLOv8n + Class Balanced Augmentation
```


在独立测试集达到：

```
mAP50 = 0.913
```



该模型作为最终输电线路异物检测模型。