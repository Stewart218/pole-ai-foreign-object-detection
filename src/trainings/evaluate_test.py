from ultralytics import YOLO


# ============================
# 1. 模型路径
# ============================

model_path = (
    r"E:\Pole_AI_Project\src\trainings\runs\detect\runs\train\week3_small_object_aug\weights\best.pt"
)




# ============================
# 2. 数据集yaml
# ============================

data_yaml = (
    r"E:\Pole_AI_Project\data\processed\dataset.yaml"
)



# ============================
# 3. 加载模型
# ============================

model = YOLO(model_path)



# ============================
# 4. test集验证
# ============================

results = model.val(

    data=data_yaml,

    split="test",

    imgsz=640,

    batch=16,

    device=0

)



# ============================
# 5. 输出指标
# ============================


print("==============================")
print("Evaluation Result")
print("==============================")


print(
    "Precision:",
    results.box.mp
)


print(
    "Recall:",
    results.box.mr
)


print(
    "mAP50:",
    results.box.map50
)


print(
    "mAP50-95:",
    results.box.map
)