# Pole AI Project 环境配置说明


## 1. 项目环境概述


本项目基于 YOLOv8 深度学习目标检测框架，实现输电线路异物检测。


主要开发环境：

|项目|配置|
|-|-|
|操作系统|Windows 11|
|Python|3.9.25|
|深度学习框架|PyTorch|
|目标检测框架|Ultralytics YOLOv8|
|GPU加速|CUDA|
|开发工具|PyCharm|



---

# 2. 硬件环境


|硬件|参数|
|-|-|
|CPU|Intel Core系列|
|GPU|NVIDIA GeForce RTX 4060 Laptop GPU|
|显存|8GB|
|CUDA Capability|8.9|



---

# 3. 软件环境


## Python环境

Python 3.9.25


## Conda环境


环境名称：
pole_ai39


环境路径：
E:\Anaconda3\envs\pole_ai39


---

# 4. 深度学习环境


## PyTorch


版本：
torch 2.8.0+cu126

CUDA支持：
CUDA Version: 12.6


检测GPU：


```python
import torch


print(torch.cuda.is_available())

print(torch.cuda.get_device_name(0))
```

输出：

```text
True

NVIDIA GeForce RTX 4060 Laptop GPU
```

# 5. 核心Python依赖

YOLO框架: 
```text
ultralytics==8.4.89
```

用途：
    模型训练
    模型验证
    推理预测
    指标评估

图像处理:
```text
opencv-python
Pillow
```

用途：
    图像读取
    标注可视化
    数据增强

数据处理：
```text
numpy
pandas
pyyaml
scikit-learn
tqdm
```

用途：
    数据统计
    数据划分
    配置文件解析


# 6. 环境创建方式

## 方法1：Anaconda推荐方式

进入项目目录：

```bash
cd E:\Pole_AI_Project
```

创建环境：

```bash
conda env create -f environment.yml
```

激活：

```bash
conda activate pole_ai39
```


## 方法2：pip安装方式

创建Python环境：

```bash
conda create -n pole_ai39 python=3.9
```

进入环境：

```bash
conda activate pole_ai39
```

安装依赖：

```bash
pip install -r requirements.txt
```


# 7. 环境验证


## 7.1 验证YOLO
执行：

```bash
yolo version
```

或者：

```python
from ultralytics import YOLO

print("YOLO environment OK")
```

## 7.2 验证GPU

运行：

```python
import torch


print(torch.cuda.is_available())

print(torch.cuda.get_device_name())
```

预期结果：
```text
True

NVIDIA GeForce RTX 4060 Laptop GPU
```


# 8. 当前最终环境版本记录

```text
Python              3.9.25

PyTorch             2.8.0+cu126

CUDA                12.6

Ultralytics         8.4.89

OpenCV              4.x

NumPy               2.x

Pandas              2.x
```


# 9. 注意事项

## 9.1 GPU版本匹配

如果重新部署：
必须安装支持CUDA的PyTorch版本。
推荐：
torch==2.8.0+cu126

## 9.2 Windows多进程问题

若运行DataLoader出现：

```text
RuntimeError:
An attempt has been made to start a new process
```

请：
    设置workers=0；
或添加：

```text
if __name__=="__main__":
    main()
```

## 9.3 路径要求

项目全部采用相对路径。
运行脚本时：
请保证当前目录：

```text
Pole_AI_Project
```



