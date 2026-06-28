"""
web_app.py — Flask Web 界面，上传图片即返回垃圾分类检测结果。
"""

import os
import io
import base64
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# ── 全局配置 ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))

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

# ── 初始化 Flask 和模型 ───────────────────────────────────────────────────
app = Flask(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INFO] 模型加载完成")


def get_garbage_label(coco_class_id: int):
    if coco_class_id in COCO_TO_GARBAGE:
        return COCO_TO_GARBAGE[coco_class_id]
    return None


def detect_and_annotate(image_np: np.ndarray):
    """对图像执行检测并返回标注后的图像和统计结果。"""
    results = model(image_np, conf=CONF_THRESHOLD)
    detected = {}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id))

            mapped = get_garbage_label(cls_id)
            if mapped is None:
                continue

            garbage_name, color = mapped
            label = f"{garbage_name} ({cls_name}) {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_np, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image_np, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detected.setdefault(garbage_name, 0)
            detected[garbage_name] += 1

    return image_np, detected


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 读取图片
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_np is None:
        return jsonify({"error": "无法解析图片"}), 400

    # 保存原始图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_path = os.path.join(OUTPUT_DIR, f"upload_{timestamp}.jpg")
    cv2.imwrite(orig_path, image_np)

    # 检测
    annotated, detected = detect_and_annotate(image_np.copy())

    # 保存标注图片
    anno_path = os.path.join(OUTPUT_DIR, f"detected_{timestamp}.jpg")
    cv2.imwrite(anno_path, annotated)

    # 编码为 base64 返回
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "detected": detected,
        "total_objects": sum(detected.values()),
        "image_base64": img_base64,
        "saved_path": anno_path,
    })


@app.route("/output/<path:filename>")
def download_file(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)