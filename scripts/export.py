"""
YOLOv8 模型导出脚本
作者：周楚为

支持导出格式：
- ONNX：跨平台推理
- TensorRT：NVIDIA GPU加速
- OpenVINO：Intel硬件加速
- CoreML：Apple设备
- TFLite：移动端部署
"""

from ultralytics import YOLO
import os
from pathlib import Path


def export_model(
    model_path: str = "runs/train/smoking_yolov8s/weights/best.pt",
    export_format: str = "onnx",
    img_size: int = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
):
    """
    导出 YOLOv8 模型
    
    Args:
        model_path: 模型权重路径
        export_format: 导出格式（onnx, torchscript, tensorrt, openvino等）
        img_size: 输入图像尺寸
        half: 是否使用FP16
        dynamic: 是否使用动态输入尺寸
        simplify: 是否简化ONNX模型
    """
    
    print("=" * 50)
    print("YOLOv8 模型导出")
    print("=" * 50)
    print(f"源模型：{model_path}")
    print(f"导出格式：{export_format}")
    print(f"输入尺寸：{img_size}")
    print("=" * 50)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 导出模型
    export_path = model.export(
        format=export_format,
        imgsz=img_size,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
    )
    
    print("\n" + "=" * 50)
    print(f"模型已导出到：{export_path}")
    print("=" * 50)
    
    return export_path


def export_all_formats(model_path: str):
    """导出所有常用格式"""
    
    formats = [
        ("onnx", "ONNX格式 - 跨平台推理"),
        ("torchscript", "TorchScript - PyTorch部署"),
    ]
    
    print("开始导出所有格式...")
    
    for fmt, desc in formats:
        print(f"\n正在导出 {desc}...")
        try:
            export_model(model_path, fmt)
        except Exception as e:
            print(f"导出 {fmt} 失败：{e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 模型导出")
    parser.add_argument("--model", type=str,
                        default="runs/train/smoking_yolov8s/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "tensorrt", "openvino",
                                "coreml", "tflite", "paddle"],
                        help="导出格式")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--half", action="store_true",
                        help="使用FP16精度")
    parser.add_argument("--dynamic", action="store_true",
                        help="动态输入尺寸")
    parser.add_argument("--all", action="store_true",
                        help="导出所有格式")
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if args.all:
        export_all_formats(args.model)
    else:
        export_model(
            model_path=args.model,
            export_format=args.format,
            img_size=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
        )
