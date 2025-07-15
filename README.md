# Car Counter YOLO
A Python project that uses YOLOv8 for real-time vehicle detection and counting via camera or a video.

# 🚗 Real-Time Vehicle Counter using YOLOv8, OpenCV & SORT

> ⚡ A robust, real-time vehicle detection, tracking, and counting system built with YOLOv8, OpenCV, and the SORT algorithm. Designed for traffic surveillance, smart city analytics, and intelligent transportation systems.

---

![Demo](Assets/demo.gif) <!-- Replace with an actual demo GIF or image -->

---

## 📌 Features

- 🔍 **YOLOv8-based object detection** (cars, buses, trucks, motorbikes)
- 🎯 **Accurate tracking** using the SORT (Simple Online and Realtime Tracking) algorithm
- 🧠 **Region of Interest (ROI)** masking to filter detections
- 🧮 **Vehicle counting** logic based on line-crossing
- 🎨 **Overlay graphics** using transparent PNGs
- 🎥 Compatible with **videos and webcams**
- 🧰 Easy to **configure, extend, and deploy**

---

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```
git clone https://github.com/Sravan2804/Car-Counter-YOLO.git
cd Car-Counter-YOLO
```

### 2️⃣ Set up a virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

## ▶️ Running the Application
Run with a video file:
```
python main.py
```
By default, the code uses:
```
cap = cv.VideoCapture("Assets/cars.mp4")
```
To use a webcam instead:
```
cap = cv.VideoCapture(0)
```

## 🧠 Tech Stack
| Tool/Library | Purpose                             |
| ------------ | ----------------------------------- |
| YOLOv8       | Real-time object detection          |
| OpenCV       | Frame handling, drawing, processing |
| cvzone       | Simplified UI overlays on OpenCV    |
| SORT         | Object tracking (ID assignment)     |
| NumPy        | Array math and matrix ops           |
| Python 3.8+  | Programming language                |

## Use Cases
- 🚦 Smart traffic monitoring
- 📈 Traffic flow analytics
- 🚧 Toll booth vehicle counting
- 🅿️ Parking management systems
- 🎥 CCTV-based traffic insights
- 🏙️ Smart city dashboards


## 🧰 Future Improvements
- Direction-aware IN/OUT counting
- Log results to CSV/Excel
- Streamlit or Gradio Web UI
- Export video with bounding boxes
- FPS & system resource monitor
- Multi-lane, multi-zone tracking













