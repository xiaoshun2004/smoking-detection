"""
YOLOv8 吸烟检测模型验证脚本
作者：周楚为
"""

from ultralytics import YOLO
import os
from pathlib import Path


def validate(
    model_path: str = "runs/train/smoking_yolov8s/weights/best.pt",
    data_config: str = "configs/smoking.yaml",
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0"
):
    """
    验证 YOLOv8 吸烟检测模型
    
    Args:
        model_path: 模型权重路径
        data_config: 数据集配置文件
        batch_size: 批次大小
        img_size: 输入图像尺寸
        device: 设备
    """
    
    print("=" * 50)
    print("YOLOv8 吸烟检测模型验证")
    print("=" * 50)
    print(f"模型：{model_path}")
    print(f"数据集配置：{data_config}")
    print("=" * 50)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 运行验证
    results = model.val(
        data=data_config,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        split="val",
        save_json=True,
        plots=True,
        verbose=True,
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("验证结果：")
    print("=" * 50)
    print(f"mAP@0.5      : {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {results.box.map:.4f}")
    print(f"Precision    : {results.box.mp:.4f}")
    print(f"Recall       : {results.box.mr:.4f}")
    print("=" * 50)
    
    # 分类别结果
    print("\n各类别 AP@0.5：")
    for i, (name, ap) in enumerate(zip(results.names.values(), results.box.ap50)):
        print(f"  {name}: {ap:.4f}")
    
    return results


if __name__ == "__main__":
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 运行验证
    validate()
