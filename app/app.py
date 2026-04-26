"""
基于 YOLOv8 的吸烟行为检测系统 — Streamlit 展示界面
作者：周楚为
启动：cd smoking-detection && streamlit run app/app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import csv
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# WebRTC（可选，用于更流畅的摄像头实时监控）
try:
    import av  # type: ignore
    from streamlit_webrtc import (  # type: ignore
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase,
    )

    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

MODEL_CANDIDATES = [
    PROJECT_ROOT / "runs" / "detect" / "train5" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "train" / "smoking_yolov8s" / "weights" / "best.pt",
]

EXAMPLE_DIR = PROJECT_ROOT / "examples"


# ---------------------------------------------------------------------------
# 模型加载（缓存，只加载一次）
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return YOLO(str(p)), str(p)
    st.error("未找到训练好的模型权重，请检查 runs/ 目录")
    st.stop()


# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="吸烟行为检测系统",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 自定义样式
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2.2rem; }
    .metric-card p  { margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.9; }
    .warn-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        color: #856404;
    }
    .safe-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px 16px;
        border-radius: 4px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🚭 吸烟行为检测系统")
    st.caption("基于 YOLOv8 的毕业设计项目")
    st.divider()

    page = st.radio(
        "功能导航",
        ["项目概览", "图片检测", "视频检测", "摄像头监控"],
        index=0,
    )

    st.divider()
    st.markdown("**作者：** 周楚为")
    st.markdown("**课题：** 基于 YOLOv8 的吸烟行为检测技术研究")


# ===================================================================
# 页面一：项目概览
# ===================================================================
if page == "项目概览":
    st.title("基于 YOLOv8 的吸烟行为检测系统")
    st.markdown("---")

    # ---------- 技术路线 ----------
    st.subheader("技术路线")
    cols = st.columns(5)
    steps = [
        ("1️⃣", "数据集构建", "Roboflow 853 张\nTrain/Val/Test"),
        ("2️⃣", "模型训练", "YOLOv8n 微调\n50 epochs"),
        ("3️⃣", "模型评估", "测试集验证\n对比实验"),
        ("4️⃣", "预警策略", "稳态触发\n冷却+留证"),
        ("5️⃣", "系统展示", "Streamlit\n图片/视频检测"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"### {icon} {title}")
            st.caption(desc)

    st.markdown("---")

    # ---------- 核心指标 ----------
    st.subheader("模型核心指标（测试集）")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown('<div class="metric-card"><h2>92.2%</h2><p>mAP@0.5</p></div>',
                    unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h2>56.0%</h2><p>mAP@0.5:0.95</p></div>',
                    unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h2>94.1%</h2><p>Precision</p></div>',
                    unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><h2>85.9%</h2><p>Recall</p></div>',
                    unsafe_allow_html=True)
    with m5:
        st.markdown('<div class="metric-card"><h2>6.4ms</h2><p>推理速度/图</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ---------- 对比实验 ----------
    st.subheader("对比实验")
    st.markdown("验证领域适配微调的有效性：通用预训练模型 vs 本课题微调模型")

    import pandas as pd
    df = pd.DataFrame({
        "模型": ["YOLOv8n（预训练）", "YOLOv8s（预训练）", "本课题模型（微调）"],
        "mAP@0.5": [0.005, 0.004, 0.922],
        "mAP@0.5:0.95": [0.003, 0.002, 0.560],
        "Precision": [0.010, 0.008, 0.941],
        "Recall": [0.116, 0.071, 0.859],
        "推理速度(ms)": [6.6, 8.8, 6.4],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.info("通用预训练模型在吸烟检测数据集上 mAP 不到 0.01，几乎完全不可用；"
            "经过本课题数据集微调后，mAP@0.5 提升至 0.922，验证了迁移学习与领域适配训练的必要性。")

    st.markdown("---")

    # ---------- 预警策略 ----------
    st.subheader("预警策略设计")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        | 机制 | 说明 |
        |------|------|
        | 连续帧稳态触发 | 连续 3 帧检测到才报警，过滤单帧误检 |
        | 冷却机制 | 报警后 5 秒内不重复报警 |
        | 报警留证 | 触发时自动保存带检测框的截图 |
        | 事件日志 | 时间戳、帧号、置信度、截图路径写入 CSV |
        """)
    with p2:
        st.markdown("""
        **设计目的：**
        - 降低误报率（单帧误检不触发）
        - 防止重复报警（冷却期内抑制）
        - 保留证据链（截图 + 结构化日志）
        - 满足监控场景的工程化需求
        """)

    st.markdown("---")

    # ---------- 评估可视化 ----------
    st.subheader("评估可视化（测试集）")
    st.caption("以下结果由 `yolo val` 在测试集上跑出，对应 `runs/detect/final_test_report/`")

    eval_dir = PROJECT_ROOT / "runs" / "detect" / "final_test_report"
    pr_path = eval_dir / "BoxPR_curve.png"
    f1_path = eval_dir / "BoxF1_curve.png"
    cm_path = eval_dir / "confusion_matrix.png"

    if eval_dir.exists():
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            if pr_path.exists():
                st.image(str(pr_path), caption="PR 曲线（Precision-Recall）",
                         use_container_width=True)
                st.caption("曲线下面积即 mAP@0.5 = 0.922。曲线越靠右上角，模型性能越好。")
        with ev_col2:
            if f1_path.exists():
                st.image(str(f1_path), caption="F1 曲线（不同置信度阈值下的 F1 值）",
                         use_container_width=True)
                st.caption("F1 是 Precision 和 Recall 的调和平均。曲线峰值 0.90 对应最佳置信度阈值 0.481。")

        if cm_path.exists():
            _, cm_col, _ = st.columns([1, 2, 1])
            with cm_col:
                st.image(str(cm_path), caption="混淆矩阵",
                         use_container_width=True)
                st.caption("行=预测，列=真实。**TP=179**（正确检出）、**FN=19**（漏检）、"
                           "**FP=22**（误报）。测试集共 198 个真实吸烟实例。")
    else:
        st.info(f"未找到评估结果目录 `{eval_dir.relative_to(PROJECT_ROOT)}`，"
                "请先执行 `yolo val model=runs/detect/train5/weights/best.pt "
                "data=configs/smoking.yaml split=test imgsz=640 name=final_test_report`。")

    st.markdown("---")

    # ---------- 环境信息 ----------
    st.subheader("实验环境")
    env = pd.DataFrame({
        "项目": ["操作系统", "Python", "Ultralytics", "PyTorch", "GPU", "显存"],
        "版本/型号": ["Windows 10", "3.10", "8.4.5", "2.7.1+cu118",
                     "NVIDIA RTX 3050", "8 GB"],
    })
    st.dataframe(env, use_container_width=True, hide_index=True)


# ===================================================================
# 页面二：图片检测
# ===================================================================
elif page == "图片检测":
    st.title("图片检测")
    st.markdown("上传一张图片，系统将自动检测其中的吸烟行为。")
    st.markdown("---")

    model, model_path = load_model()
    st.sidebar.success(f"模型已加载：{Path(model_path).name}")

    # 参数设置
    with st.sidebar:
        st.markdown("### 检测参数")
        conf_thresh = st.slider("置信度阈值", 0.10, 0.90, 0.25, 0.05,
                                help="越高越严格，可能漏检；越低可能误检")
        iou_thresh = st.slider("IoU 阈值（NMS）", 0.10, 0.90, 0.45, 0.05,
                               help="非极大值抑制阈值，用于过滤重叠框")

    # 图片来源
    upload_tab, example_tab = st.tabs(["上传图片", "示例图片"])

    input_image = None

    with upload_tab:
        uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded is not None:
            input_image = Image.open(uploaded).convert("RGB")

    with example_tab:
        if EXAMPLE_DIR.exists():
            examples = sorted(EXAMPLE_DIR.glob("*.*"))
            examples = [e for e in examples if e.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            if examples:
                cols = st.columns(min(len(examples), 4))
                for i, ex in enumerate(examples):
                    with cols[i % 4]:
                        img = Image.open(ex).convert("RGB")
                        st.image(img, caption=ex.name, use_container_width=True)
                        if st.button(f"使用此图", key=f"ex_{i}"):
                            input_image = img
            else:
                st.info("examples/ 目录下暂无图片")
        else:
            st.info("未找到 examples/ 目录，可创建该目录并放入示例图片")

    # 执行检测
    if input_image is not None:
        st.markdown("---")

        with st.spinner("正在检测..."):
            img_array = np.array(input_image)
            results = model.predict(source=img_array, conf=conf_thresh, iou=iou_thresh, verbose=False)
            result = results[0]
            result_img = result.plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**原始图片**")
            st.image(input_image, use_container_width=True)
        with col_result:
            st.markdown("**检测结果**")
            st.image(result_img, use_container_width=True)

        # 检测信息
        boxes = result.boxes
        if len(boxes) > 0:
            st.markdown('<div class="warn-box">⚠️ <b>检测到吸烟行为！</b></div>',
                        unsafe_allow_html=True)
            st.markdown("")
            for box in boxes:
                cls_name = result.names[int(box.cls[0])]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                st.markdown(f"- **{cls_name}**  置信度 `{conf_val:.2%}`  "
                            f"位置 ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
        else:
            st.markdown('<div class="safe-box">✅ 未检测到吸烟行为</div>',
                        unsafe_allow_html=True)


# ===================================================================
# 页面三：视频检测
# ===================================================================
elif page == "视频检测":
    st.title("视频检测")
    st.markdown("上传一段视频，系统将逐帧检测吸烟行为并生成标注后的结果视频。")
    st.markdown("---")

    model, model_path = load_model()
    st.sidebar.success(f"模型已加载：{Path(model_path).name}")

    with st.sidebar:
        st.markdown("### 检测参数")
        conf_thresh = st.slider("置信度阈值 ", 0.10, 0.90, 0.45, 0.05,
                                key="vid_conf")
        iou_thresh = st.slider("IoU 阈值 ", 0.10, 0.90, 0.45, 0.05,
                               key="vid_iou")
        stable_frames = st.number_input("稳态触发帧数", 1, 10, 5,
                                        help="连续 N 帧检测到才记录为一次预警事件")
        cooldown_sec = st.number_input("冷却时间（秒）", 1, 30, 10,
                                       help="两次预警之间的最小间隔")
        save_evidence_vid = st.toggle("保存报警留证（截图+CSV）", value=True,
                                      help="触发预警时保存带检测框截图到 runs/detect/events/，并写入 event_log.csv")

    uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # 保存上传的视频到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        input_path = tfile.name

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(f"视频信息：{w}x{h}，{fps:.1f} fps，共 {total_frames} 帧")

        if st.button("开始检测", type="primary"):
            # 输出视频路径
            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_file.name
            out_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            cap = cv2.VideoCapture(input_path)
            progress = st.progress(0, text="正在处理...")
            status_text = st.empty()

            # 预警状态
            stable_count = 0
            last_event_time = 0.0
            events = []
            frame_idx = 0

            # 留证输出（与摄像头监控共用目录）
            event_dir = PROJECT_ROOT / "runs" / "detect" / "events"
            log_path = event_dir / "event_log.csv"

            def ensure_log_header_vid():
                event_dir.mkdir(parents=True, exist_ok=True)
                if not log_path.exists():
                    with open(log_path, "w", newline="", encoding="utf-8") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(["timestamp", "source", "frame_index", "max_conf", "event_image"])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                results = model.predict(source=frame, conf=conf_thresh,
                                        iou=iou_thresh, verbose=False)
                result = results[0]
                plotted = result.plot()
                video_writer.write(plotted)

                # 预警逻辑
                boxes = result.boxes
                if len(boxes) > 0:
                    max_conf = float(boxes.conf.max().item())
                    stable_count += 1
                else:
                    max_conf = 0.0
                    stable_count = 0

                now = time.time()
                if stable_count >= stable_frames and (now - last_event_time) >= cooldown_sec:
                    event_path_str = ""
                    if save_evidence_vid:
                        ensure_log_header_vid()
                        event_name = f"event_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_idx:06d}.jpg"
                        event_path = event_dir / event_name
                        cv2.imwrite(str(event_path), plotted)
                        event_path_str = str(event_path)
                        with open(log_path, "a", newline="", encoding="utf-8") as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow([datetime.now().isoformat(), "video_upload", frame_idx, f"{max_conf:.4f}", event_path_str])

                    events.append({
                        "帧号": frame_idx,
                        "时间位置": f"{frame_idx / fps:.1f}s",
                        "最高置信度": f"{max_conf:.2%}",
                        "留证截图": event_path_str if event_path_str else "(未保存)",
                    })
                    last_event_time = now
                    stable_count = 0

                # 更新进度
                pct = frame_idx / max(total_frames, 1)
                progress.progress(min(pct, 1.0),
                                  text=f"正在处理... {frame_idx}/{total_frames} 帧")
                if frame_idx % 30 == 0:
                    status_text.caption(f"已检测 {frame_idx} 帧，发现 {len(events)} 次预警事件")

            cap.release()
            video_writer.release()

            progress.progress(1.0, text="处理完成！")
            status_text.caption(f"共处理 {frame_idx} 帧，发现 {len(events)} 次预警事件")

            st.markdown("---")

            # 播放结果视频
            st.subheader("检测结果视频")
            with open(out_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)

            # 下载按钮
            st.download_button(
                label="下载结果视频",
                data=video_bytes,
                file_name="smoking_detection_result.mp4",
                mime="video/mp4",
            )

            # 预警日志
            if events:
                st.subheader("预警事件日志")
                st.markdown('<div class="warn-box">⚠️ 检测到吸烟行为预警事件</div>',
                            unsafe_allow_html=True)
                st.markdown("")
                import pandas as pd
                st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="safe-box">✅ 视频中未触发预警事件</div>',
                            unsafe_allow_html=True)

            # 清理临时文件
            try:
                os.unlink(input_path)
                os.unlink(out_path)
            except OSError:
                pass


# ===================================================================
# 页面四：摄像头监控（B2：OpenCV 采集 + Streamlit 刷新）
# ===================================================================
elif page == "摄像头监控":
    st.title("摄像头监控（实时预警）")
    st.markdown("使用本机摄像头进行实时吸烟检测，支持稳态触发、冷却机制、留证截图与事件日志。")
    st.markdown("---")

    model, model_path = load_model()
    st.sidebar.success(f"模型已加载：{Path(model_path).name}")

    # 模式选择（推荐 WebRTC，避免 B2 的 rerun 闪烁）
    mode = st.radio(
        "监控模式",
        ["WebRTC（推荐，更流畅）", "兼容模式（OpenCV 刷新）"],
        index=0 if _HAS_WEBRTC else 1,
        horizontal=True,
        help="WebRTC 模式视频不会依赖页面反复刷新，画面更稳定；兼容模式无需额外依赖，但可能闪烁。",
    )

    # 侧边栏参数
    with st.sidebar:
        st.markdown("### 摄像头参数")
        cam_index = st.selectbox("摄像头编号", options=[0, 1, 2, 3], index=0,
                                 help="一般内置摄像头为 0，外接摄像头可能为 1/2/3")
        refresh_ms = st.slider("刷新间隔（毫秒）", 50, 500, 120, 10,
                               help="越小越流畅，但 CPU 压力更大")
        infer_every = st.slider("每 N 帧推理一次", 1, 10, 1, 1,
                                help="核显/CPU 环境可适当增大以降低压力")

        st.markdown("### 检测参数")
        conf_thresh = st.slider("置信度阈值", 0.10, 0.90, 0.45, 0.05,
                                key="cam_conf")
        iou_thresh = st.slider("IoU 阈值（NMS）", 0.10, 0.90, 0.45, 0.05,
                               key="cam_iou")
        stable_frames = st.number_input("稳态触发帧数", 1, 10, 5,
                                        help="连续 N 次（推理帧）检测到才触发一次预警")
        cooldown_sec = st.number_input("冷却时间（秒）", 1, 60, 10,
                                       help="两次预警之间的最小间隔")
        save_evidence = st.toggle("保存留证（截图+CSV日志）", value=True)

    # Session state 初始化
    if "cam_cap" not in st.session_state:
        st.session_state.cam_cap = None
    if "cam_frame_idx" not in st.session_state:
        st.session_state.cam_frame_idx = 0
    if "cam_stable_count" not in st.session_state:
        st.session_state.cam_stable_count = 0
    if "cam_last_event_time" not in st.session_state:
        st.session_state.cam_last_event_time = 0.0
    if "cam_events" not in st.session_state:
        st.session_state.cam_events = []
    if "cam_last_max_conf" not in st.session_state:
        st.session_state.cam_last_max_conf = 0.0
    if "cam_events_version" not in st.session_state:
        st.session_state.cam_events_version = 0
    if "cam_events_rendered_version" not in st.session_state:
        st.session_state.cam_events_rendered_version = -1

    # 控制区
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        run_cam = st.checkbox("开启摄像头监控", value=False)
    with ctrl_col2:
        if st.button("重置状态"):
            st.session_state.cam_frame_idx = 0
            st.session_state.cam_stable_count = 0
            st.session_state.cam_last_event_time = 0.0
            st.session_state.cam_last_max_conf = 0.0
            st.session_state.cam_events = []
    with ctrl_col3:
        st.caption("提示：若画面卡顿，可提高“刷新间隔”或将“每 N 帧推理一次”设为 2~5。")

    # 展示区
    # 让画面占比更小：左(画面)窄、右(状态)宽
    view_col, info_col = st.columns([1, 2])
    with view_col:
        frame_holder = st.empty()
    with info_col:
        metric_holder = st.empty()
        st.markdown("---")
        st.subheader("预警事件（本次会话）")
        events_holder = st.empty()
        st.caption("留证文件默认保存到 `runs/detect/events/`。")

    event_dir = PROJECT_ROOT / "runs" / "detect" / "events"
    log_path = event_dir / "event_log.csv"

    def ensure_log_header():
        event_dir.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "source", "frame_index", "max_conf", "event_image"])

    def do_rerun():
        # 兼容不同 Streamlit 版本
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    # ==============================
    # WebRTC 模式（推荐）
    # ==============================
    if mode.startswith("WebRTC"):
        # 如果之前使用过 OpenCV 兼容模式，确保释放摄像头句柄，避免被占用导致 WebRTC 启动失败
        cap = st.session_state.get("cam_cap")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        st.session_state.cam_cap = None

        if not _HAS_WEBRTC:
            st.warning("当前环境未安装 `streamlit-webrtc`，已自动切换到兼容模式（OpenCV 刷新）。")
            st.caption("安装命令（清华源）：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit-webrtc`")
        else:
            st.info(
                "若出现 `NotReadableError: Could not start video source`，通常是摄像头被占用或权限被系统阻止。"
                "请先关闭微信/QQ/钉钉/浏览器其他网页等占用摄像头的软件，并在 Windows「设置→隐私与安全→相机」"
                "中确认允许应用访问相机；然后刷新页面重试。"
            )
            import threading

            class SmokingWebRTCProcessor(VideoProcessorBase):
                def __init__(self):
                    self._lock = threading.Lock()
                    self.frame_idx = 0
                    self.stable_count = 0
                    self.last_event_time = 0.0
                    self.last_max_conf = 0.0
                    self.events = []  # 最新在前

                def _ensure_log_header(self):
                    event_dir.mkdir(parents=True, exist_ok=True)
                    if not log_path.exists():
                        with open(log_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["timestamp", "source", "frame_index", "max_conf", "event_image"])

                def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                    img = frame.to_ndarray(format="bgr24")

                    with self._lock:
                        self.frame_idx += 1
                        idx = self.frame_idx

                    plotted = img
                    max_conf_local = self.last_max_conf

                    # 按间隔推理
                    if idx % int(infer_every) == 0:
                        results = model.predict(source=img, conf=conf_thresh, iou=iou_thresh, verbose=False)
                        result = results[0]
                        plotted = result.plot()  # BGR

                        boxes = result.boxes
                        if len(boxes) > 0:
                            max_conf_local = float(boxes.conf.max().item())
                            with self._lock:
                                self.stable_count += 1
                        else:
                            max_conf_local = 0.0
                            with self._lock:
                                self.stable_count = 0

                        with self._lock:
                            self.last_max_conf = max_conf_local

                        now = time.time()
                        with self._lock:
                            can_fire = self.stable_count >= int(stable_frames) and (now - self.last_event_time) >= float(cooldown_sec)

                        if can_fire:
                            event_path_str = ""
                            if save_evidence:
                                self._ensure_log_header()
                                event_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:06d}.jpg"
                                event_path = event_dir / event_name
                                cv2.imwrite(str(event_path), plotted)
                                event_path_str = str(event_path)
                                with open(log_path, "a", newline="", encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([datetime.now().isoformat(), f"webrtc_webcam_{cam_index}", idx, f"{max_conf_local:.4f}", event_path_str])

                            with self._lock:
                                self.events.insert(0, {
                                    "时间": datetime.now().strftime("%H:%M:%S"),
                                    "帧号": idx,
                                    "最高置信度": f"{max_conf_local:.2%}",
                                    "留证截图": event_path_str if event_path_str else "(未保存)",
                                })
                                self.events = self.events[:50]
                                self.last_event_time = now
                                self.stable_count = 0

                    return av.VideoFrame.from_ndarray(plotted, format="bgr24")

            # 展示区（WebRTC）
            # 让画面占比更小：左(视频)窄、右(状态)宽
            view_col, info_col = st.columns([1, 2])
            with view_col:
                st.subheader("实时画面（WebRTC）")
                st.caption("点击下方 Start 开启摄像头。若提示权限，请允许浏览器访问摄像头。")
            with info_col:
                st.subheader("实时状态")
                metric_holder = st.empty()
                st.markdown("---")
                st.subheader("预警事件（本次会话）")
                events_holder = st.empty()
                st.caption("留证文件默认保存到 `runs/detect/events/`。")

            rtc_config = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            webrtc_ctx = webrtc_streamer(
                key="smoking_webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=SmokingWebRTCProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # 从 processor 拉取状态（无需重绘视频）
            proc = webrtc_ctx.video_processor
            if proc is not None:
                try:
                    with proc._lock:
                        idx = proc.frame_idx
                        maxc = proc.last_max_conf
                        stable = proc.stable_count
                        evs = list(proc.events)
                except Exception:
                    idx, maxc, stable, evs = 0, 0.0, 0, []

                metric_holder.markdown(
                    f"""
                    - 帧号：`{idx}`
                    - 最高置信度：`{maxc:.2%}`
                    - 稳态计数：`{stable}` / `{int(stable_frames)}`
                    - 已触发事件数：`{len(evs)}`
                    """
                )
                if evs:
                    import pandas as pd
                    events_holder.dataframe(pd.DataFrame(evs), use_container_width=True, hide_index=True)
                else:
                    events_holder.markdown('<div class="safe-box">✅ 尚未触发预警事件</div>',
                                           unsafe_allow_html=True)
            else:
                st.info("点击 Start 后开始显示实时状态与事件。")

            # WebRTC 模式到此结束，不再进入 B2
            st.stop()

    if run_cam:
        # 采集一帧（B2 模式：每次运行读一帧 + rerun 实现“准实时”）
        # 为减少闪烁：摄像头句柄在 session 内常驻，不每次打开/关闭
        cap = st.session_state.cam_cap
        if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
            try:
                cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
            except Exception:
                cap = cv2.VideoCapture(int(cam_index))
            st.session_state.cam_cap = cap

        if cap is None or not cap.isOpened():
            st.session_state.cam_cap = None
            st.error("无法打开摄像头，请检查摄像头权限或尝试切换摄像头编号。")
            st.stop()

        ok, frame = cap.read()

        if not ok or frame is None:
            # 尝试重连一次
            try:
                cap.release()
            except Exception:
                pass
            st.session_state.cam_cap = None
            st.error("读取摄像头画面失败（已尝试重连）。请重试或切换摄像头编号。")
            st.stop()

        st.session_state.cam_frame_idx += 1
        frame_idx = st.session_state.cam_frame_idx

        inferred = False
        max_conf = st.session_state.cam_last_max_conf
        plotted_bgr = frame

        # 按间隔推理
        if frame_idx % int(infer_every) == 0:
            inferred = True
            results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
            result = results[0]
            plotted_bgr = result.plot()  # BGR

            boxes = result.boxes
            if len(boxes) > 0:
                max_conf = float(boxes.conf.max().item())
                st.session_state.cam_stable_count += 1
            else:
                max_conf = 0.0
                st.session_state.cam_stable_count = 0

            st.session_state.cam_last_max_conf = max_conf

            # 预警触发
            now = time.time()
            if st.session_state.cam_stable_count >= int(stable_frames) and (now - st.session_state.cam_last_event_time) >= float(cooldown_sec):
                event_path_str = ""
                if save_evidence:
                    ensure_log_header()
                    event_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_idx:06d}.jpg"
                    event_path = event_dir / event_name
                    cv2.imwrite(str(event_path), plotted_bgr)
                    event_path_str = str(event_path)

                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().isoformat(), f"webcam_{cam_index}", frame_idx, f"{max_conf:.4f}", event_path_str])

                st.session_state.cam_events.insert(0, {
                    "时间": datetime.now().strftime("%H:%M:%S"),
                    "帧号": frame_idx,
                    "最高置信度": f"{max_conf:.2%}",
                    "留证截图": event_path_str if event_path_str else "(未保存)",
                })
                st.session_state.cam_events = st.session_state.cam_events[:50]
                st.session_state.cam_events_version += 1

                st.session_state.cam_last_event_time = now
                st.session_state.cam_stable_count = 0

        # 显示画面
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
        caption = f"摄像头 {cam_index} | 帧 {frame_idx} | " + ("推理帧" if inferred else "跳过推理")
        frame_holder.image(plotted_rgb, caption=caption, use_container_width=True)

        # 显示状态
        metric_holder.markdown(
            f"""
            **当前状态**
            - 帧号：`{frame_idx}`
            - 最高置信度：`{max_conf:.2%}`
            - 稳态计数：`{st.session_state.cam_stable_count}` / `{int(stable_frames)}`
            - 已触发事件数：`{len(st.session_state.cam_events)}`
            """
        )

        # 事件表（只在有新事件时刷新，减少重绘闪烁）
        if st.session_state.cam_events and st.session_state.cam_events_rendered_version != st.session_state.cam_events_version:
            import pandas as pd
            events_holder.dataframe(pd.DataFrame(st.session_state.cam_events),
                                    use_container_width=True, hide_index=True)
            st.session_state.cam_events_rendered_version = st.session_state.cam_events_version
        elif not st.session_state.cam_events:
            events_holder.markdown('<div class="safe-box">✅ 尚未触发预警事件</div>',
                                   unsafe_allow_html=True)

        # 刷新
        time.sleep(float(refresh_ms) / 1000.0)
        do_rerun()

    else:
        # 停止时释放摄像头
        cap = st.session_state.cam_cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        st.session_state.cam_cap = None
        st.info("勾选“开启摄像头监控”后将开始实时检测。若不想保存留证，可在侧边栏关闭保存开关。")
        metric_holder.markdown(
            f"""
            **当前状态**
            - 帧号：`{st.session_state.cam_frame_idx}`
            - 已触发事件数：`{len(st.session_state.cam_events)}`
            - 留证目录：`{event_dir.relative_to(PROJECT_ROOT)}`
            """
        )
        if st.session_state.cam_events:
            import pandas as pd
            events_holder.dataframe(pd.DataFrame(st.session_state.cam_events),
                                    use_container_width=True, hide_index=True)
        else:
            events_holder.markdown('<div class="safe-box">✅ 尚未触发预警事件</div>',
                                   unsafe_allow_html=True)
