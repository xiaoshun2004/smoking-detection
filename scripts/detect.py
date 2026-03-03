"""
YOLOv8 吸烟检测推理脚本
作者：周楚为
"""

from ultralytics import YOLO
import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2


def detect(
    source: str,
    model_path: str = "runs/train/smoking_yolov8s/weights/best.pt",
    conf: float = 0.5,
    iou: float = 0.6,
    save: bool = True,
    show: bool = False,
    device: str = "0",
    stream: bool = False,
    stable_frames: int = 3,
    cooldown_seconds: int = 5,
    event_dir: str = "runs/detect/events",
    log_path: str = "runs/detect/events/event_log.csv",
    max_det: int = 100,
):
    """
    运行 YOLOv8 吸烟检测推理
    
    Args:
        source: 输入源（图片/视频/摄像头ID/文件夹）
        model_path: 模型权重路径
        conf: 置信度阈值
        iou: NMS IoU 阈值
        save: 是否保存结果
        show: 是否显示结果
        device: 设备（'0' GPU, 'cpu' CPU）
        stream: 是否启用流式推理（视频/摄像头推荐）
        stable_frames: 连续触发帧数（稳态触发）
        cooldown_seconds: 报警冷却时间（秒）
        event_dir: 报警截图输出目录
        log_path: 报警日志 CSV 路径
        max_det: 每帧最大检测框数量
    """
    
    print("=" * 50)
    print("YOLOv8 吸烟检测推理")
    print("=" * 50)
    print(f"输入源：{source}")
    print(f"模型：{model_path}")
    print(f"置信度阈值：{conf}")
    print(f"稳态触发帧数：{stable_frames}")
    print(f"冷却时间：{cooldown_seconds}s")
    print("=" * 50)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 运行推理
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        max_det=max_det,
        save=save,
        show=show,
        device=device,
        project="runs/detect",
        name="predict",
        stream=stream,  # 视频/摄像头建议 True
    )
    
    # 预警稳态触发与日志
    if stable_frames < 1:
        stable_frames = 1
    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "source", "frame_index", "max_conf", "event_image"])

    frame_count = 0
    stable_count = 0
    last_event_time = 0.0

    # 打印结果统计 + 稳态触发
    for i, result in enumerate(results):
        frame_count += 1
        boxes = result.boxes

        max_conf = 0.0
        if len(boxes) > 0:
            max_conf = float(boxes.conf.max().item())
            stable_count += 1
        else:
            stable_count = 0

        print(f"\n图片/帧 {i + 1}:")
        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = result.names[cls]
                print(f"  - 检测到：{class_name}，置信度：{conf_val:.2f}")
        else:
            print("  - 未检测到目标")

        now = time.time()
        if stable_count >= stable_frames and (now - last_event_time) >= cooldown_seconds:
            # 保存报警截图
            event_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_count:06d}.jpg"
            event_path = os.path.join(event_dir, event_name)
            plotted = result.plot()  # BGR
            cv2.imwrite(event_path, plotted)

            # 记录日志
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), str(source), frame_count, f"{max_conf:.4f}", event_path])

            print(f"  >>> 触发报警：{event_path}")
            last_event_time = now
            stable_count = 0
    
    print("\n" + "=" * 50)
    if save:
        print("结果已保存到：runs/detect/predict")
    print(f"报警截图目录：{event_dir}")
    print(f"报警日志：{log_path}")
    print("=" * 50)
    
    return results


def detect_webcam(model_path: str = "runs/train/smoking_yolov8s/weights/best.pt"):
    """使用摄像头实时检测（带稳态触发）"""
    detect(
        source="0",
        model_path=model_path,
        conf=0.5,
        iou=0.6,
        save=False,
        show=True,
        device="0",
        stream=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 吸烟检测推理")
    parser.add_argument("--source", type=str, default="data/images/test",
                        help="输入源：图片/视频/摄像头ID/文件夹")
    parser.add_argument("--model", type=str, 
                        default="runs/train/smoking_yolov8s/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU 阈值")
    parser.add_argument("--save", action="store_true", default=True,
                        help="保存结果")
    parser.add_argument("--show", action="store_true",
                        help="显示结果")
    parser.add_argument("--device", type=str, default="0",
                        help="设备：0 (GPU) 或 cpu")
    parser.add_argument("--webcam", action="store_true",
                        help="使用摄像头")
    parser.add_argument("--stream", action="store_true",
                        help="启用流式推理（视频/摄像头推荐）")
    parser.add_argument("--stable-frames", type=int, default=3,
                        help="稳态触发所需连续帧数")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="报警冷却时间（秒）")
    parser.add_argument("--event-dir", type=str, default="runs/detect/events",
                        help="报警截图保存目录")
    parser.add_argument("--log-path", type=str, default="runs/detect/events/event_log.csv",
                        help="报警日志 CSV 路径")
    parser.add_argument("--max-det", type=int, default=100,
                        help="每帧最大检测框数量")
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if args.webcam:
        detect_webcam(args.model)
    else:
        detect(
            source=args.source,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show,
            device=args.device,
            stream=args.stream,
            stable_frames=args.stable_frames,
            cooldown_seconds=args.cooldown,
            event_dir=args.event_dir,
            log_path=args.log_path,
            max_det=args.max_det,
        )
