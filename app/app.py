"""
YOLOv8 吸烟检测 Web 演示界面
作者：周楚为

使用 Gradio 构建简洁的 Web 界面
"""

import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from pathlib import Path


# 模型路径
MODEL_PATH = "runs/train/smoking_yolov8s/weights/best.pt"

# 全局模型对象
model = None


def load_model():
    """加载模型"""
    global model
    if model is None:
        # 检查模型是否存在
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            # 使用预训练模型作为演示
            print(f"警告：未找到训练好的模型 {MODEL_PATH}")
            print("使用预训练模型 yolov8s.pt 作为演示...")
            model = YOLO("yolov8s.pt")
    return model


def detect_image(image, conf_threshold, iou_threshold):
    """
    检测图片中的吸烟行为
    
    Args:
        image: 输入图片
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
    
    Returns:
        检测结果图片, 检测信息文本
    """
    if image is None:
        return None, "请上传图片"
    
    # 加载模型
    model = load_model()
    
    # 运行推理
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    
    # 获取结果
    result = results[0]
    
    # 绘制检测框
    result_image = result.plot()
    result_image = Image.fromarray(result_image[..., ::-1])  # BGR to RGB
    
    # 生成检测信息
    boxes = result.boxes
    if len(boxes) > 0:
        info_lines = ["🔍 检测结果：\n"]
        
        # 统计各类别数量
        class_counts = {}
        for box in boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            conf_val = float(box.conf[0])
            
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(conf_val)
        
        for class_name, confs in class_counts.items():
            avg_conf = sum(confs) / len(confs)
            info_lines.append(f"• {class_name}: {len(confs)} 个 (平均置信度: {avg_conf:.2%})")
        
        # 预警判断
        if "smoking" in class_counts:
            info_lines.append("\n⚠️ 警告：检测到吸烟行为！")
        
        info_text = "\n".join(info_lines)
    else:
        info_text = "✅ 未检测到吸烟行为"
    
    return result_image, info_text


def create_interface():
    """创建 Gradio 界面"""
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        font-family: 'Microsoft YaHei', sans-serif;
    }
    .title {
        text-align: center;
        color: #2c3e50;
    }
    """
    
    with gr.Blocks(css=custom_css, title="吸烟检测系统") as demo:
        gr.Markdown(
            """
            # 🚬 基于 YOLOv8 的吸烟行为检测系统
            
            上传图片即可检测吸烟行为，支持调整检测参数。
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                input_image = gr.Image(
                    label="📤 上传图片",
                    type="pil",
                    height=400,
                )
                
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="置信度阈值",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        label="IoU 阈值",
                    )
                
                detect_btn = gr.Button("🔍 开始检测", variant="primary")
            
            with gr.Column(scale=1):
                # 输出区域
                output_image = gr.Image(
                    label="📊 检测结果",
                    height=400,
                )
                output_text = gr.Textbox(
                    label="📝 检测信息",
                    lines=6,
                )
        
        # 示例图片
        gr.Markdown("### 📷 示例图片")
        gr.Examples(
            examples=[
                ["examples/example1.jpg"],
                ["examples/example2.jpg"],
            ],
            inputs=[input_image],
            outputs=[output_image, output_text],
            fn=lambda x: detect_image(x, 0.25, 0.45),
            cache_examples=False,
        ) if os.path.exists("examples") else None
        
        # 绑定事件
        detect_btn.click(
            fn=detect_image,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_text],
        )
        
        gr.Markdown(
            """
            ---
            
            **说明：**
            - 置信度阈值：越高检测越严格，可能漏检；越低可能误检
            - IoU 阈值：用于非极大值抑制，过滤重叠框
            
            **作者：** 周楚为 | 华南理工大学软件学院
            """
        )
    
    return demo


if __name__ == "__main__":
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为True可生成公网链接
    )
