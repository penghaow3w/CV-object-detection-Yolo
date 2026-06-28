"""
detect_realtime.py — 实时视频/摄像头物体检测（垃圾分类场景）
使用 YOLOv8 + COCO 预训练权重，对视频流或摄像头进行实时检测。
"""

import argparse
import os
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO

# ── COCO 类别 → 垃圾分类映射 ──────────────────────────────────────────────
GARBAGE_CATEGORIES = {
    "塑料": {"coco_ids": [39], "color": (0, 165, 255)},
    "纸张": {"coco_ids": [73], "color": (0, 255, 255)},
    "金属": {"coco_ids": [42, 43, 44, 76], "color": (192, 192, 192)},
    "玻璃": {"coco_ids": [40, 41, 75], "color": (255, 0, 0)},
    "厨余": {"coco_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55], "color": (0, 255, 0)},
}

COCO_TO_GARBAGE = {}
for cat_name, info in GARBAGE_CATEGORIES.items():
    for cid in info["coco_ids"]:
        COCO_TO_GARBAGE[cid] = (cat_name, info["color"])


def get_garbage_label(coco_class_id: int):
    if coco_class_id in COCO_TO_GARBAGE:
        return COCO_TO_GARBAGE[coco_class_id]
    return None


def draw_boxes(frame, boxes_data, model_names):
    """在帧上绘制边界框和标签。"""
    for box in boxes_data:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        mapped = get_garbage_label(cls_id)
        if mapped is None:
            continue

        garbage_name, color = mapped
        cls_name = model_names.get(cls_id, str(cls_id))
        label = f"{garbage_name} ({cls_name}) {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def draw_legend(frame):
    """在画面左上角绘制垃圾分类图例。"""
    y_start = 30
    x = 10
    for cat_name, info in GARBAGE_CATEGORIES.items():
        color = info["color"]
        cv2.rectangle(frame, (x, y_start), (x + 20, y_start + 16), color, -1)
        cv2.putText(frame, cat_name, (x + 28, y_start + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 22
    return frame


def detect_realtime(source, model_path="yolov8n.pt", conf=0.25, save_video=False,
                    output_dir="output"):
    """实时视频/摄像头检测。"""
    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)
    model_names = model.names

    # 确定视频源
    if source.isdigit():
        source = int(source)
        print(f"[INFO] 打开摄像头: {source}")
    else:
        if not os.path.exists(source):
            print(f"[ERROR] 视频文件不存在: {source}")
            sys.exit(1)
        print(f"[INFO] 打开视频文件: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频源: {source}")
        sys.exit(1)

    # 视频写入器
    writer = None
    if save_video:
        os.makedirs(output_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"realtime_detected_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"[INFO] 视频将保存至: {out_path}")

    print("[INFO] 开始实时检测，按 'q' 键退出...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 视频流结束")
            break

        frame_count += 1
        # 每 2 帧检测一次以提高性能
        if frame_count % 2 == 0:
            results = model(frame, conf=conf, verbose=False)
            if results and results[0].boxes is not None:
                frame = draw_boxes(frame, results[0].boxes, model_names)
        else:
            # 使用上一次的检测结果继续绘制
            pass

        frame = draw_legend(frame)

        if writer:
            writer.write(frame)

        cv2.imshow("YOLOv8 垃圾分类实时检测 (按 q 退出)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] 检测结束")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 垃圾分类实时检测")
    parser.add_argument("--source", default="0",
                        help="视频源: 摄像头索引(默认0) 或 视频文件路径")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径 (默认: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--save", action="store_true", help="保存检测结果视频")
    parser.add_argument("--output", default="output", help="输出目录 (默认: output)")
    args = parser.parse_args()

    detect_realtime(args.source, args.model, args.conf,
                    save_video=args.save, output_dir=args.output)


if __name__ == "__main__":
    main()