"""
detect_image.py — 图片物体检测（垃圾分类场景）
使用 YOLOv8 + COCO 预训练权重，对图片中的垃圾进行分类检测。
"""

import argparse
import os
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO

# ── COCO 类别 → 垃圾分类映射 ──────────────────────────────────────────────
GARBAGE_CATEGORIES = {
    # 塑料 (Plastic)
    "塑料": {"coco_ids": [39], "color": (0, 165, 255)},  # bottle (橙色)
    # 纸张 (Paper)
    "纸张": {"coco_ids": [73], "color": (0, 255, 255)},  # book (黄色)
    # 金属 (Metal)
    "金属": {"coco_ids": [42, 43, 44, 76], "color": (192, 192, 192)},  # fork, knife, spoon, scissors (银色)
    # 玻璃 (Glass)
    "玻璃": {"coco_ids": [40, 41, 75], "color": (255, 0, 0)},  # wine glass, cup, vase (蓝色)
    # 厨余 (Kitchen Waste)
    "厨余": {"coco_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55], "color": (0, 255, 0)},  # food items (绿色)
}

# 构建 COCO ID → 垃圾类别 的反向映射
COCO_TO_GARBAGE = {}
for cat_name, info in GARBAGE_CATEGORIES.items():
    for cid in info["coco_ids"]:
        COCO_TO_GARBAGE[cid] = (cat_name, info["color"])


def get_garbage_label(coco_class_id: int, coco_class_name: str):
    """将 COCO 类别映射到垃圾分类类别，非垃圾类别返回 None。"""
    if coco_class_id in COCO_TO_GARBAGE:
        garbage_name, color = COCO_TO_GARBAGE[coco_class_id]
        return garbage_name, color
    return None


def draw_results(image, results):
    """在图片上绘制检测结果：边界框 + 垃圾分类标签 + 置信度。"""
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id))

            mapped = get_garbage_label(cls_id, cls_name)
            if mapped is None:
                continue

            garbage_name, color = mapped
            label = f"{garbage_name} ({cls_name}) {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # 标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def detect_image(image_path: str, model_path: str = "yolov8n.pt", conf: float = 0.25,
                 save: bool = True, output_dir: str = "output"):
    """对单张图片执行检测。"""
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片不存在: {image_path}")
        sys.exit(1)

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] 检测图片: {image_path}")
    results = model(image_path, conf=conf)

    # 读取图片并绘制结果
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        sys.exit(1)

    annotated = draw_results(image, results)

    detected = {}
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = result.names.get(cls_id, str(cls_id))
            mapped = get_garbage_label(cls_id, cls_name)
            if mapped:
                cat = mapped[0]
                detected.setdefault(cat, 0)
                detected[cat] += 1

    print("\n===== 检测结果 =====")
    if detected:
        for cat, count in detected.items():
            print(f"  {cat}类垃圾: {count} 个")
    else:
        print("  未检测到垃圾类物品")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{fname}_detected_{timestamp}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"\n[INFO] 结果已保存至: {out_path}")

    return annotated, detected


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 垃圾分类图片检测")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径 (默认: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--no-save", action="store_true", help="不保存检测结果图片")
    parser.add_argument("--output", default="output", help="输出目录 (默认: output)")
    parser.add_argument("--show", action="store_true", help="显示检测结果窗口")
    args = parser.parse_args()

    annotated, _ = detect_image(args.image, args.model, args.conf,
                                save=not args.no_save, output_dir=args.output)

    if args.show:
        cv2.imshow("YOLOv8 垃圾分类检测", annotated)
        print("[INFO] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()