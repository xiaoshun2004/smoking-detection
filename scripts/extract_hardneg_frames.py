"""
从视频中均匀抽帧，并生成硬负样本空标签（YOLO txt）。

用法（在项目根目录）：
  conda activate yolo
  python scripts/extract_hardneg_frames.py --input-dir data/hardneg --count 20

默认输出：
  data/hardneg/images/*.jpg
  data/hardneg/labels/*.txt  (空文件，表示无目标)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def linspace_indices(total: int, count: int) -> list[int]:
    """生成 [0, total-1] 上均匀分布的 count 个帧索引（去重、升序）。"""
    if total <= 0:
        return []
    if count <= 1:
        return [0]
    # 手写 linspace，避免额外依赖 numpy
    out: list[int] = []
    for i in range(count):
        x = int(round(i * (total - 1) / (count - 1)))
        out.append(max(0, min(total - 1, x)))
    # 去重保持顺序
    dedup: list[int] = []
    last = None
    for x in out:
        if last is None or x != last:
            dedup.append(x)
        last = x
    return dedup


def extract_frames(video_path: Path, out_images: Path, out_labels: Path, count: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        # 兜底：顺序读一遍计数（短视频可接受）
        total_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法重新打开视频：{video_path}")

    idxs = linspace_indices(total_frames, count)
    saved = 0
    stem = video_path.stem

    for k, frame_idx in enumerate(idxs, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        img_name = f"{stem}_{k:03d}.jpg"
        img_path = out_images / img_name
        ok2 = cv2.imwrite(str(img_path), frame)
        if not ok2:
            continue

        # 生成空标签
        label_path = out_labels / f"{stem}_{k:03d}.txt"
        label_path.write_text("", encoding="utf-8")
        saved += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="包含视频文件的目录")
    parser.add_argument("--count", type=int, default=20, help="每个视频抽取帧数（默认 20）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"目录不存在：{input_dir}")

    out_images = input_dir / "images"
    out_labels = input_dir / "labels"
    ensure_dir(out_images)
    ensure_dir(out_labels)

    videos = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        raise SystemExit(f"未在目录中找到视频文件：{input_dir}")

    total_saved = 0
    for vp in videos:
        saved = extract_frames(vp, out_images, out_labels, int(args.count))
        total_saved += saved
        print(f"[OK] {vp.name}: 保存 {saved} 帧 -> {out_images}")

    print(f"\n完成：共保存 {total_saved} 张图片，空标签已生成于 {out_labels}")


if __name__ == "__main__":
    main()

