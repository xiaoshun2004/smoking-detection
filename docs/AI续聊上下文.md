# AI 续聊上下文（回家继续用）

> 目的：在另一台电脑打开本仓库时，让 AI 通过阅读此文件快速恢复上下文，直接继续推进项目与论文写作。
>
> 使用方式：回家打开 Cursor 后，对我说“先读 `docs/AI续聊上下文.md`，然后继续做 XXX”。

## 1. 项目定位（一句话）
基于 **YOLOv8 单类目标检测（Smoking）** 的吸烟行为检测与安全预警系统：支持图片/视频/摄像头输入，采用 **稳态触发 + 冷却机制 + 留证截图 + CSV 事件日志** 的工程化预警闭环，并提供 Streamlit Web 演示界面。

## 2. 代码与功能现状（以本仓库 `smoking-detection-deploy` 为准）

### 2.1 Web 演示（Streamlit）
- **启动入口**：`run_app.py`
- **主界面文件**：`app/app.py`
- **页面**：
  - 项目概览
  - 图片检测
  - 视频检测（已支持：触发预警时保存留证截图 + 写入 `event_log.csv`）
  - 摄像头监控（两种模式）：
    - **B1 WebRTC**：更流畅，但可能出现浏览器摄像头权限/占用导致的 `NotReadableError`。
    - **B2 OpenCV 兼容模式**：无需额外依赖，但靠 `st.rerun()` 刷新，可能轻微闪烁；已做“摄像头句柄常驻”减少闪烁。

### 2.2 命令行推理与预警
- `scripts/detect.py`：视频/摄像头推理 + 稳态触发 + 冷却 + 留证 + 日志（工程闭环）

### 2.3 数据集配置
- `configs/smoking.yaml`：已改为 **相对路径** `path: data/dataset`，避免写死 G 盘路径。
  - 期望结构：`data/dataset/{train,valid,test}/{images,labels}`
  - 注意：仓库默认不包含数据集（`.gitignore` 忽略），需自行从 Roboflow 下载解压放入上述目录。

## 3. 预警留证输出（重要，论文/答辩证据链）
- **留证截图目录**：`runs/detect/events/`（默认忽略 `*.jpg`，避免仓库膨胀）
- **事件日志**：`runs/detect/events/event_log.csv`
- **日志字段**：`timestamp, source, frame_index, max_conf, event_image`

## 4. 当前推荐默认参数（已写入前端默认值）
面向“减少玩手机/拿笔等相似动作误报”的演示优先参数：
- `conf=0.45`
- `stable_frames=5`
- `cooldown=10s`

> 说明：`stable_frames` 过大（如 8）会显著降低误报，但会增加漏报与报警延迟；论文中可做对比实验（3/5/8）。

## 5. 已修复/已做的重要改动点（便于回溯）
1) **视频检测留证**：`app/app.py` 的“视频检测”页面新增开关“保存报警留证（截图+CSV）”，触发预警会写入 `runs/detect/events/` 与 `event_log.csv`。  
2) **writer 命名冲突修复**：视频检测中将 `cv2.VideoWriter` 与 `csv.writer` 分别改名为 `video_writer` / `csv_writer`，避免 `AttributeError: '_csv.writer' object has no attribute 'write'`。  
3) **摄像头监控画面占比**：布局调整为左窄右宽（`st.columns([1, 2])`）。  
4) **WebRTC 失败提示**：进入 WebRTC 模式会释放 OpenCV 摄像头句柄，减少占用冲突，并提示权限/占用排查。  
5) **依赖**：`requirements.txt` 已加入 `streamlit-webrtc`（B1 模式依赖）。

## 6. 硬负样本（Hard Negative）改进实验（未完成，回家台式优先做）
### 6.1 现象
模型易将“白色细长物体”（笔/吸管等）误报为香烟（Smoking）。

### 6.2 已完成
- 新增脚本：`scripts/extract_hardneg_frames.py`
  - 作用：从 `data/hardneg/` 中的视频均匀抽帧，输出：
    - `data/hardneg/images/*.jpg`
    - `data/hardneg/labels/*.txt`（空标签，表示负样本无目标）

### 6.3 下一步（回家台式机做）
- 将 hardneg 图片+空标签 **并入训练集**（建议只并入 train）并续训：
  - 续训权重：`runs/detect/train5/weights/last.pt`
  - 对比：补前 vs 补后 的误报次数/预警触发次数（同一视频素材/同一参数）

## 7. 基线对比实验计划（未完成，回家台式优先做）
- 目标：**YOLOv8s 微调**作为基线，对比当前 YOLOv8n 微调模型
- 口径：同数据集、同 split=test、同 imgsz，报告 `mAP50/mAP50-95/P/R/速度`。

## 8. 论文写作状态（已完成/待写）
### 8.1 已完成
- 中英文摘要（含定量指标）
- 第1章 绪论
- 第3章 数据集构建与标注（Roboflow 数据集：853 张，train 600 / val 173 / test 80）
- 第4章 方法（YOLOv8 检测 + 预警策略）

### 8.2 待写（建议顺序）
1) 第6章 系统实现与演示（**不要写死 UI 细节**，重点写模块与输入输出、留证与日志）
2) 第5章 实验与结果分析（先写已有结果 + 误差分析框架；回家补基线/硬负样本实验数据后再完善）
3) 第2章 相关技术与研究现状（最后补）
4) 第7章 总结与展望

## 9. 回家环境复现（命令清单）
### 9.1 安装与启动
```bat
cd /d <repo_root>
conda create -n yolo python=3.10 -y
conda activate yolo
python -m pip install -r requirements.txt
python run_app.py
```

### 9.2 可选依赖（WebRTC 更流畅）
```bat
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit-webrtc
```

### 9.3 训练/评估（台式机）
```bat
yolo val model="runs/detect/train5/weights/best.pt" data="configs/smoking.yaml" split=test imgsz=640 name=final_test_report
```

