# CV Object Detection - YOLO

基于 YOLOv8 的实时垃圾分类检测系统 | Real-time Garbage Classification Detection System based on YOLOv8

[English](#english) | [中文](#中文)

---

## English

### Overview

A real-time object detection system built with **Ultralytics YOLOv8** and **COCO pretrained weights**, specialized for garbage classification. It supports 5 garbage categories and provides three modes: image detection, real-time video/camera detection, and a Flask web interface.

### Features

- **YOLOv8 + COCO Pretrained**: High-accuracy object detection out of the box
- **5 Garbage Categories**: Plastic, Paper, Metal, Glass, Kitchen Waste
- **3 Detection Modes**:
  - `detect_image.py` — Single image detection
  - `detect_realtime.py` — Real-time video / webcam detection
  - `web_app.py` — Flask web interface with drag-and-drop upload
- **Rich Annotations**: Bounding boxes + garbage category label + COCO class name + confidence score
- **Local Saving**: Detection results automatically saved to `output/` directory

### Garbage Category Mapping

| Garbage Category | COCO Classes Mapped | BBox Color |
|:---:|:---|:---:|
| **Plastic** (塑料) | bottle (39) | Orange |
| **Paper** (纸张) | book (73) | Yellow |
| **Metal** (金属) | fork, knife, spoon, scissors (42, 43, 44, 76) | Silver |
| **Glass** (玻璃) | wine glass, cup, vase (40, 41, 75) | Blue |
| **Kitchen Waste** (厨余) | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake (46–55) | Green |

### Project Structure

```
cv-object-detection/
├── detect_image.py          # CLI image detection
├── detect_realtime.py       # CLI real-time video/camera detection
├── web_app.py               # Flask web server
├── templates/
│   └── index.html           # Web UI (upload + results)
├── requirements.txt         # Python dependencies
├── output/                  # Saved detection results (gitignored)
└── .gitignore
```

### Quick Start

**1. Clone and install dependencies**

```bash
git clone https://github.com/penghaow3w/CV-object-detection-Yolo.git
cd cv-object-detection
pip install -r requirements.txt
```

**2. Download the model (auto-downloaded on first run, or manually)**

```bash
# The model yolov8n.pt will be auto-downloaded from Ultralytics on first use.
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
```

**3. Run**

```bash
# Image detection
python detect_image.py path/to/image.jpg

# Image detection with custom threshold
python detect_image.py path/to/image.jpg --conf 0.5

# Real-time camera detection (default device 0)
python detect_realtime.py --source 0

# Video file detection with save
python detect_realtime.py --source video.mp4 --save

# Start web server
python web_app.py
# Open http://localhost:5000 in your browser
```

### CLI Arguments

#### detect_image.py

| Argument | Default | Description |
|:---|:---|:---|
| `image` | *(required)* | Input image path |
| `--model` | `yolov8n.pt` | YOLO model path |
| `--conf` | `0.25` | Confidence threshold |
| `--no-save` | — | Disable saving results |
| `--output` | `output` | Output directory |
| `--show` | — | Show result window |

#### detect_realtime.py

| Argument | Default | Description |
|:---|:---|:---|
| `--source` | `0` | Camera index or video file path |
| `--model` | `yolov8n.pt` | YOLO model path |
| `--conf` | `0.25` | Confidence threshold |
| `--save` | — | Save output video |
| `--output` | `output` | Output directory |

### Web API

**POST /detect** — Upload an image for detection

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/detect
```

Response:

```json
{
  "detected": {"塑料": 1, "金属": 2},
  "total_objects": 3,
  "image_base64": "...",
  "saved_path": "output/detected_20260628_120000.jpg"
}
```

### Requirements

- Python >= 3.8
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- flask >= 3.0.0
- pillow >= 10.0.0
- numpy >= 1.24.0

---

## 中文

### 概述

基于 **Ultralytics YOLOv8** 和 **COCO 预训练权重** 构建的实时垃圾分类检测系统。支持 5 类垃圾识别，提供图片检测、实时视频/摄像头检测和 Flask Web 界面三种使用模式。

### 功能特点

- **YOLOv8 + COCO 预训练**：开箱即用的高精度检测
- **5 类垃圾分类**：塑料、纸张、金属、玻璃、厨余
- **3 种检测模式**：
  - `detect_image.py` — 单张图片检测
  - `detect_realtime.py` — 实时视频 / 摄像头检测
  - `web_app.py` — Flask Web 界面，拖拽上传
- **丰富标注信息**：边界框 + 垃圾分类标签 + COCO 类别名 + 置信度
- **本地保存**：检测结果自动保存至 `output/` 目录

### 垃圾分类映射表

| 垃圾类别 | 映射的 COCO 类别 | 边框颜色 |
|:---:|:---|:---:|
| **塑料** | bottle (39) | 橙色 |
| **纸张** | book (73) | 黄色 |
| **金属** | fork, knife, spoon, scissors (42, 43, 44, 76) | 银色 |
| **玻璃** | wine glass, cup, vase (40, 41, 75) | 蓝色 |
| **厨余** | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake (46–55) | 绿色 |

### 项目结构

```
cv-object-detection/
├── detect_image.py          # 命令行图片检测
├── detect_realtime.py       # 命令行实时视频/摄像头检测
├── web_app.py               # Flask Web 服务
├── templates/
│   └── index.html           # Web 前端界面（上传 + 结果展示）
├── requirements.txt         # Python 依赖
├── output/                  # 检测结果保存目录（已 gitignore）
└── .gitignore
```

### 快速开始

**1. 克隆仓库并安装依赖**

```bash
git clone https://github.com/penghaow3w/CV-object-detection-Yolo.git
cd cv-object-detection
pip install -r requirements.txt
```

**2. 下载模型（首次运行自动下载，也可手动下载）**

```bash
# 首次运行时 yolov8n.pt 会自动从 Ultralytics 下载。
# 或手动下载：
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
```

**3. 运行**

```bash
# 图片检测
python detect_image.py path/to/image.jpg

# 图片检测（自定义置信度阈值）
python detect_image.py path/to/image.jpg --conf 0.5

# 实时摄像头检测（默认设备 0）
python detect_realtime.py --source 0

# 视频文件检测并保存结果
python detect_realtime.py --source video.mp4 --save

# 启动 Web 服务
python web_app.py
# 浏览器访问 http://localhost:5000
```

### 命令行参数

#### detect_image.py

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `image` | *(必填)* | 输入图片路径 |
| `--model` | `yolov8n.pt` | YOLO 模型路径 |
| `--conf` | `0.25` | 置信度阈值 |
| `--no-save` | — | 不保存检测结果 |
| `--output` | `output` | 输出目录 |
| `--show` | — | 显示检测结果窗口 |

#### detect_realtime.py

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--source` | `0` | 摄像头索引 或 视频文件路径 |
| `--model` | `yolov8n.pt` | YOLO 模型路径 |
| `--conf` | `0.25` | 置信度阈值 |
| `--save` | — | 保存检测结果视频 |
| `--output` | `output` | 输出目录 |

### Web API

**POST /detect** — 上传图片进行检测

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/detect
```

返回示例：

```json
{
  "detected": {"塑料": 1, "金属": 2},
  "total_objects": 3,
  "image_base64": "...",
  "saved_path": "output/detected_20260628_120000.jpg"
}
```

### 环境要求

- Python >= 3.8
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- flask >= 3.0.0
- pillow >= 10.0.0
- numpy >= 1.24.0

### 注意事项

1. 模型权重 `yolov8n.pt` 约 6.3MB，首次运行会自动下载，需要网络连接
2. 本系统基于 COCO 预训练权重的类别映射实现垃圾分类，非专门训练的垃圾分类模型，检测精度取决于 COCO 类别与垃圾物品的匹配程度
3. 如需更高精度，建议使用专门的垃圾分类数据集对 YOLO 模型进行微调（fine-tune）
4. `detect_realtime.py` 需要摄像头设备或视频文件，在无 GUI 环境中需使用 `--save` 参数保存结果