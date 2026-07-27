from ultralytics import YOLO


def main():

    model_path = (
        r"E:\Pole_AI_Project\runs\train\week3_aug_v2\weights\best.pt"
    )


    data_yaml = (
        r"E:\Pole_AI_Project\data\processed\dataset.yaml"
    )


    model = YOLO(model_path)


    results = model.val(

        data=data_yaml,

        split="test",

        imgsz=640,

        batch=8,

        workers=0,

        device=0

    )


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



if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()