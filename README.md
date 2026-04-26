# 基于 YOLOv8 的吸烟行为检测系统

## 项目简介

本项目为毕业设计课题，基于 YOLOv8（detect）实现监控场景中的吸烟行为检测与预警系统。
支持图片/视频/摄像头输入，具备"连续帧稳态触发 + 冷却机制 + 报警留证 + 日志记录"的工程化预警策略。

## 环境信息

| 项目 | 版本/型号 |
|---|---|
| OS | Windows 10 |
| Python | 3.10 |
| Ultralytics | 8.4.5 |
| PyTorch | 2.7.1+cu118 |
| GPU | NVIDIA GeForce RTX 3050 8GB |
| Conda 环境名 | yolo |

## 项目结构

```
smoking-detection/
├── configs/
│   └── smoking.yaml            # 数据集配置（指向 Roboflow 导出目录）
├── scripts/
│   ├── train.py                # 训练脚本
│   ├── val.py                  # 验证脚本
│   ├── detect.py               # 推理 + 预警脚本（稳态触发 + 日志）
│   └── export.py               # 模型导出脚本
├── app/
│   └── app.py                  # Web 界面（可选）
├── docs/
│   ├── 实验记录模板.md          # 实验记录（含对比表格、指标、路径）
│   ├── 外文翻译_原文.txt        # 外文翻译原文
│   └── 外文翻译_译文.txt        # 外文翻译译文
├── runs/
│   └── detect/
│       ├── train5/weights/         # 训练好的模型权重
│       │   ├── best.pt             # 最优权重（用于推理/展示）
│       │   └── last.pt             # 最后一轮权重（用于续训）
│       ├── final_test_report/      # 测试集评估结果（PR 曲线、混淆矩阵等）
│       └── events/                 # 预警截图 + 日志
│           └── event_log.csv
├── data/                           # 数据集占位（不上传，见下方说明）
├── weights/                        # 权重占位
├── run_app.py                      # 一键启动 Streamlit Web 界面
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始（在新电脑上恢复环境）

### 1. 克隆项目
```bash
git clone https://github.com/你的用户名/smoking-detection.git
cd smoking-detection
```

### 2. 创建 conda 环境
```bash
conda create -n yolo python=3.10 -y
conda activate yolo
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 准备数据集
数据集不随项目上传（太大）。请从 Roboflow 下载后放到指定目录：
- 数据集链接：https://universe.roboflow.com/truong-jbsna/detect-smoking-behavior
- 下载格式：YOLOv8
- 放置路径示例：`G:\桌面\毕业设计\detect smoking behavior.v6i.yolov8\`
- 然后修改 `configs/smoking.yaml` 中的 `path` 字段指向该目录

### 5. 安装 PyTorch（GPU 版）
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 常用命令

### 训练
```bash
yolo train model=yolov8n.pt data="configs/smoking.yaml" epochs=100 imgsz=640 batch=8 workers=0
```

### 续训
```bash
yolo train model="runs/detect/train5/weights/last.pt" data="configs/smoking.yaml" epochs=100 imgsz=640 batch=8 workers=0
```

### 验证/测试
```bash
yolo val model="runs/detect/train5/weights/best.pt" data="configs/smoking.yaml" split=test imgsz=640
```

### 推理（展示）
```bash
yolo predict model="runs/detect/train5/weights/best.pt" source="测试图片目录" conf=0.25 iou=0.25 save=True
```

### 预警演示（视频，带稳态触发 + 日志）
```bash
python scripts/detect.py --source "视频路径.mp4" --model "runs/detect/train5/weights/best.pt" --conf 0.25 --iou 0.25 --stable-frames 3 --cooldown 5 --stream
```

## 启动 Web 演示界面（Streamlit）

包含 **项目概览 / 图片检测 / 视频检测** 三个页面，支持上传图片和视频，自动运行模型并展示结果。

### 一键启动
```cmd
cd /d G:\grad_des\smoking-detection
conda activate yolo
python run_app.py
```

启动成功后，浏览器自动打开 `http://localhost:8501`。停止服务按 `Ctrl + C`。

### 首次使用需要安装的额外依赖
```cmd
pip install streamlit pandas
```
（如果跑 `python run_app.py` 时报 `ModuleNotFoundError`，安装对应的包即可）

### 常见问题
| 报错 | 原因 | 解决 |
|---|---|---|
| `cd` 命令不切换盘符 | Anaconda Prompt (CMD) 不能跨盘 | 使用 `cd /d G:\...` |
| `未找到训练好的模型权重` | `best.pt` 不存在 | 确认 `runs/detect/train5/weights/best.pt` 存在 |
| `ModuleNotFoundError: streamlit` | 未安装 Streamlit | `pip install streamlit` |
| 端口 8501 被占用 | 上次未正常退出 | 修改 `run_app.py` 第 18 行端口号 |

## 测试集评估（复现指标）

复现 README 中实验结果表格的指标：
```cmd
yolo val model=runs/detect/train5/weights/best.pt data=configs/smoking.yaml split=test imgsz=640 name=final_test_report
```

输出结果保存在 `runs/detect/final_test_report/`，包含：
- `BoxPR_curve.png` —— PR 曲线（曲线下面积即 mAP@0.5）
- `BoxF1_curve.png` —— F1 曲线
- `confusion_matrix.png` / `confusion_matrix_normalized.png` —— 混淆矩阵
- `val_batch*_pred.jpg` —— 验证集预测结果对比图

终端会打印关键指标（与下方"实验结果"表对应）：
```
Class     Images  Instances  Box(P     R    mAP50  mAP50-95)
all          80        198    0.941  0.859  0.922   0.560
```

## 实验结果（测试集）

| 模型 | mAP50 | mAP50-95 | Precision | Recall | 推理(ms/img) |
|---|---:|---:|---:|---:|---:|
| YOLOv8n（预训练） | 0.005 | 0.003 | 0.010 | 0.116 | 6.6 |
| YOLOv8s（预训练） | 0.004 | 0.002 | 0.008 | 0.071 | 8.8 |
| **本课题模型（微调）** | **0.922** | **0.560** | **0.941** | **0.859** | **6.4** |

## 数据集信息

| 项目 | 说明 |
|---|---|
| 名称 | detect smoking behavior v6 (Roboflow) |
| 来源 | https://universe.roboflow.com/truong-jbsna/detect-smoking-behavior |
| 规模 | train 600 / val 173 / test 80 |
| 类别 | Smoking（单类） |
| 格式 | YOLOv8 (txt) |

## 参考资料

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Roboflow 数据集](https://universe.roboflow.com/truong-jbsna/detect-smoking-behavior)

## 作者信息

- 姓名：周楚为
- 课题：基于 YOLOv8 的吸烟行为检测技术研究
