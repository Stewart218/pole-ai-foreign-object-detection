"""
Week3 Task5

Evaluate All Models

统一评估所有实验模型
生成evaluation_report_v2.txt
"""


from ultralytics import YOLO
from pathlib import Path
import time



# ============================
# 项目路径
# ============================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)



DATASET_YAML = (
    PROJECT_ROOT
    /
    "data"
    /
    "final_dataset"
    /
    "dataset.yaml"
)



# 测试集

TEST_SPLIT = "test"



# ============================
# 模型列表
# ============================


MODELS = {


"Baseline_YOLOv8n":

PROJECT_ROOT
/
"runs"
/
"train"
/
"baseline_yolov8n"
/
"weights"
/
"best.pt",



"YOLOv8s":

PROJECT_ROOT
/
"runs"
/
"train"
/
"week3_yolov8s"
/
"weights"
/
"best.pt",



"AdamW_YOLOv8n":

PROJECT_ROOT
/
"runs"
/
"train"
/
"week3_lr_opt"
/
"weights"
/
"best.pt",



"YOLOv8n_Class Balanced Augmentation":

PROJECT_ROOT
/
"runs"
/
"train"
/
"week3_aug_v2"
/
"weights"
/
"best.pt"

}



# ============================
# 输出
# ============================


OUTPUT_DIR = (

PROJECT_ROOT

/

"results"

/

"week3"

)





REPORT_FILE = (

OUTPUT_DIR

/

"evaluation_report.txt"

)




# ============================
# 评估函数
# ============================


def evaluate_model(name, weight):


    print("\n")

    print("="*60)

    print(name)

    print("="*60)



    model = YOLO(weight)



    start=time.time()



    result = model.val(

        data=str(DATASET_YAML),

        split=TEST_SPLIT,

        imgsz=640,

        device=0,

        verbose=False

    )



    cost=time.time()-start



    metrics={


        "precision":

        float(result.box.mp),



        "recall":

        float(result.box.mr),



        "mAP50":

        float(result.box.map50),



        "mAP50-95":

        float(result.box.map),



        "time":

        cost


    }



    return metrics





# ============================
# 主程序
# ============================


def main():


    OUTPUT_DIR.mkdir(
        exist_ok=True
    )



    results={}



    for name,path in MODELS.items():


        if not path.exists():

            print(
                f"模型不存在:{path}"
            )

            continue



        results[name]=evaluate_model(

            name,

            str(path)

        )




    # =========================
    # 写报告
    # =========================


    with open(

        REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as f:



        f.write(

"""
========================================
Model Evaluation Report
========================================

测试数据:
processed/split/test

输入尺寸:
640


"""
        )



        for name,data in results.items():


            f.write("\n")

            f.write(

                "="*40+"\n"

            )

            f.write(

                name+"\n"

            )

            f.write(

                "="*40+"\n"

            )


            for k,v in data.items():


                f.write(

                    f"{k}: {v:.4f}\n"

                )



        f.write("\n\n")






    print()

    print(
        "Evaluation finished!"
    )

    print(
        REPORT_FILE
    )



if __name__=="__main__":

    main()