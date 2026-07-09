from ultralytics import YOLO

# 加载官方预训练模型
model = YOLO("yolov8n.pt")

# 推理一张图片
results = model(
    r"E:\Pole_AI_Project\data\raw\pole_foreign_objects\鸟巢\bird_nest_001_54981f2d.png",
    save=True,
    conf=0.25
)

print("Inference Finished!")