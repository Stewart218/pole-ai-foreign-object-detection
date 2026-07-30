# Evaluation Metrics（目标检测评估指标说明）

## 1. 文档简介

在目标检测任务中，仅依靠模型是否能够检测到目标无法全面评价模型性能，还需要通过一系列标准化评估指标衡量模型的检测能力。本项目采用 YOLOv8 目标检测算法，重点关注 Precision（精确率）、Recall（召回率）、IoU（交并比）、AP（Average Precision）和 mAP（Mean Average Precision）等指标。

本文件介绍各评价指标的定义、计算方法、工程意义以及 YOLOv8 内置评估接口的使用方法，为后续模型训练与性能分析提供理论依据。

---

# 2. IoU（Intersection over Union，交并比）

## 2.1 定义

IoU（Intersection over Union）表示**预测边界框（Prediction Box）与真实边界框（Ground Truth Box）之间的重叠程度**。

计算公式：

```text
IoU = 预测框与真实框的交集面积 / 预测框与真实框的并集面积
```

数学表达式：

```text
IoU = Area(Prediction ∩ GroundTruth)
      --------------------------------
      Area(Prediction ∪ GroundTruth)
```

IoU 的取值范围为：

```text
0 ≤ IoU ≤ 1
```

其中：

* IoU = 1：预测框与真实框完全重合；
* IoU = 0：预测框与真实框没有任何重叠。

---

## 2.2 工程意义

IoU 用于判断模型检测是否正确。

通常规定：

| IoU       | 判断                  |
| --------- | ------------------- |
| IoU ≥ 0.5 | 检测正确（True Positive） |
| IoU < 0.5 | 检测错误                |

因此，IoU 是计算 Precision、Recall、AP 和 mAP 的基础。

---

# 3. Precision（精确率）

## 3.1 定义

Precision 表示：

**模型预测出的目标中，有多少是真正正确的目标。**

计算公式：

```text
Precision = TP / (TP + FP)
```

其中：

| 符号 | 含义                     |
| -- | ---------------------- |
| TP | True Positive（真正例）     |
| FP | False Positive（假正例、误检） |

---

## 3.2 示例

假设模型检测出了：

```text
共检测100个目标
```

其中：

```text
90个正确
10个错误
```

则：

```text
Precision = 90 / (90 + 10)

          = 90%
```

---

## 3.3 工程意义

Precision 越高说明：

* 模型误检越少；
* 检测结果越可靠；
* 巡检人员需要人工复核的目标越少。

对于输电线路巡检而言，高 Precision 可以减少误报警，提高巡检效率。

---

# 4. Recall（召回率）

## 4.1 定义

Recall 表示：

**所有真实目标中，有多少被模型成功检测出来。**

计算公式：

```text
Recall = TP / (TP + FN)
```

其中：

| 符号 | 含义                 |
| -- | ------------------ |
| TP | 真正例                |
| FN | False Negative（漏检） |

---

## 4.2 示例

真实存在：

```text
100个鸟巢
```

模型检测到：

```text
80个
```

则：

```text
Recall = 80 / 100

       = 80%
```

---

## 4.3 工程意义

Recall 越高说明：

* 漏检目标越少；
* 检测覆盖率越高；
* 巡检安全性越高。

对于输电线路异物检测而言，漏检可能导致安全隐患，因此 Recall 是重要指标。

---

# 5. Precision 与 Recall 的关系

Precision 和 Recall 往往存在一定的权衡关系。

例如：

提高检测阈值（Confidence）：

* Precision 提高；
* Recall 降低。

降低检测阈值：

* Recall 提高；
* Precision 降低。

因此，仅使用 Precision 或 Recall 无法全面评价模型性能，需要综合考虑。

---

# 6. PR 曲线（Precision-Recall Curve）

PR 曲线表示：

随着 Confidence 阈值不断变化，

Precision 与 Recall 的变化关系。

横轴：

```text
Recall
```

纵轴：

```text
Precision
```

优秀模型的 PR 曲线应尽量靠近右上角。

YOLOv8 在训练完成后会自动生成 PR 曲线图。

---

# 7. AP（Average Precision）

## 7.1 定义

AP（Average Precision）表示：

**单个类别 PR 曲线下的面积（Area Under Curve）。**

AP 越大，

说明：

该类别检测效果越好。

例如：

```text
鸟巢：

AP = 0.93
```

说明：

模型检测鸟巢效果很好。

---

# 8. mAP（Mean Average Precision）

## 8.1 定义

mAP 为：

所有类别 AP 的平均值。

例如：

四个类别：

```text
鸟巢

气球

塑料袋

其他异物
```

若：

```text
AP

0.95

0.91

0.89

0.93
```

则：

```text
mAP=(0.95+0.91+0.89+0.93)/4=0.92
```

---

## 8.2 mAP@0.5

表示：

IoU ≥ 0.5

认为检测正确。

这是目标检测最经典评价指标。

YOLO 默认重点展示：

```text
mAP50
```

即：

```text
mAP@0.5
```

---

## 8.3 mAP@0.5:0.95

YOLOv8 默认还会计算：

```text
mAP50-95
```

即：

IoU

从：

```text
0.50

0.55

0.60

……

0.95
```

每隔 0.05

计算一次 AP。

最后取平均。

该指标更加严格，

也是 COCO 数据集官方评价标准。

---

# 9. 常用评价指标总结

| 指标           | 含义          | 越大越好吗 | 工程意义      |
| ------------ | ----------- | ----- | --------- |
| IoU          | 预测框与真实框重叠程度 | 是     | 判断检测是否正确  |
| Precision    | 检测结果准确率     | 是     | 减少误检      |
| Recall       | 检测覆盖率       | 是     | 减少漏检      |
| AP           | 单类别检测能力     | 是     | 衡量单类别性能   |
| mAP@0.5      | 多类别平均检测性能   | 是     | 最常用指标     |
| mAP@0.5:0.95 | 更严格综合评价     | 是     | COCO 官方标准 |

---

# 10. YOLOv8 内置评估接口

YOLOv8 提供了自动模型评估接口，可直接计算 Precision、Recall、mAP 等指标。

加载训练完成后的模型：

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
```

开始评估：

```python
metrics = model.val()
```

YOLO 会自动完成：

* 数据集读取；
* 模型推理；
* IoU 计算；
* Precision 计算；
* Recall 计算；
* AP 计算；
* mAP 计算。

---

# 11. 常用指标读取接口

```python
print(metrics.box.map)
```

输出：

```text
mAP@0.5:0.95
```

---

```python
print(metrics.box.map50)
```

输出：

```text
mAP@0.5
```

---

```python
print(metrics.box.map75)
```

输出：

```text
mAP@0.75
```

---

```python
print(metrics.box.mp)
```

输出：

```text
Precision
```

---

```python
print(metrics.box.mr)
```

输出：

```text
Recall
```

---

# 12. YOLOv8 自动生成的评估结果

训练完成后，YOLOv8 会在：

```text
runs/detect/train/
```

目录下自动生成：

* results.png（训练指标变化曲线）
* PR_curve.png（Precision-Recall 曲线）
* F1_curve.png（F1 曲线）
* P_curve.png（Precision 曲线）
* R_curve.png（Recall 曲线）
* confusion_matrix.png（混淆矩阵）

这些图像可直接用于模型性能分析和实验报告。

---

# 13. 本项目评估策略

结合输电线路异物检测项目特点，后续模型训练将重点关注以下指标：

1. **mAP@0.5**：作为模型整体检测性能的主要评价指标；
2. **Precision**：降低误检，减少巡检人员的无效复核工作；
3. **Recall**：减少漏检，避免输电线路异物未被发现造成安全隐患；
4. **mAP@0.5:0.95**：综合评价模型在不同 IoU 阈值下的检测能力。

项目将在每次训练结束后使用 YOLOv8 的 `model.val()` 接口自动计算上述指标，并结合 PR 曲线、混淆矩阵等可视化结果，对模型性能进行分析，为后续模型优化提供依据。

---

# 14. 总结

IoU、Precision、Recall、AP 和 mAP 是目标检测中最重要的性能评价指标。其中，IoU 用于衡量预测框与真实框的重叠程度，Precision 和 Recall 分别反映模型的误检率和漏检率，AP 用于评价单类别检测性能，mAP 则综合反映整个模型的检测能力。

YOLOv8 已集成完整的评估流程，可通过 `model.val()` 自动计算各项指标，并生成多种评估图表。本项目将以 **mAP@0.5** 为主要评价指标，同时兼顾 Precision、Recall 和 mAP@0.5:0.95，以满足输电线路异物检测任务对轻量化、高精度和高可靠性的要求。
