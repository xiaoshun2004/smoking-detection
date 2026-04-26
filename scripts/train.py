"""
YOLOv8 吸烟检测模型训练脚本
作者：周楚为
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train():
    """训练 YOLOv8 吸烟检测模型"""
    
    # ==================== 配置参数 ====================
    # 模型选择（可选：yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt）
    MODEL = "yolov8s.pt"  # 推荐从 s 版本开始，平衡速度和精度
    
    # 数据集配置文件路径
    DATA_CONFIG = "configs/smoking.yaml"
    
    # 训练参数
    EPOCHS = 100          # 训练轮数
    BATCH_SIZE = 16       # 批次大小，根据显存调整
    IMG_SIZE = 640        # 输入图像尺寸
    DEVICE = 0            # GPU设备ID，使用CPU则设为 'cpu'
    
    # 输出目录
    PROJECT = "runs/train"
    NAME = "smoking_yolov8s"
    
    # ==================== 开始训练 ====================
    print("=" * 50)
    print("开始训练 YOLOv8 吸烟检测模型")
    print("=" * 50)
    
    # 加载预训练模型
    model = YOLO(MODEL)
    
    # 开始训练
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        
        # 优化器设置
        optimizer="auto",     # 自动选择优化器
        lr0=0.01,            # 初始学习率
        lrf=0.01,            # 最终学习率比例
        
        # 数据增强
        augment=True,
        hsv_h=0.015,         # 色调增强
        hsv_s=0.7,           # 饱和度增强
        hsv_v=0.4,           # 明度增强
        degrees=0.0,         # 旋转角度
        translate=0.1,       # 平移比例
        scale=0.5,           # 缩放比例
        shear=0.0,           # 剪切角度
        perspective=0.0,     # 透视变换
        flipud=0.0,          # 上下翻转概率
        fliplr=0.5,          # 左右翻转概率
        mosaic=1.0,          # Mosaic增强概率
        mixup=0.0,           # Mixup增强概率
        
        # 其他设置
        patience=50,         # 早停耐心值
        save=True,           # 保存检查点
        save_period=-1,      # 保存周期（-1表示只保存最佳和最后）
        val=True,            # 训练时验证
        plots=True,          # 生成训练曲线图
        verbose=True,        # 详细输出
    )
    
    print("=" * 50)
    print("训练完成！")
    print(f"最佳模型保存在：{PROJECT}/{NAME}/weights/best.pt")
    print("=" * 50)
    
    return results


def train_with_resume(checkpoint_path: str):
    """从检查点恢复训练"""
    model = YOLO(checkpoint_path)
    results = model.train(resume=True)
    return results


if __name__ == "__main__":
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 开始训练
    train()
