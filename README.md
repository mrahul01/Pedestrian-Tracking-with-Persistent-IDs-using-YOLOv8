# 🚶 Pedestrian Tracking with Persistent IDs using YOLOv8

A real-time computer vision project that detects and tracks pedestrians in video streams using **YOLOv8** and **OpenCV**.  
The system assigns stable, human-readable IDs to each person and visualizes their movement through trajectory trails.

---

## 🔍 Key Features
- 👤 Pedestrian detection using **YOLOv8 (COCO person class)**
- 🆔 Custom ID assignment after stable detection
- 🧭 Motion trajectory visualization using trail lines
- 🎥 Works on recorded surveillance-style video footage
- ⚡ Real-time and lightweight implementation

---

## 🛠️ Tech Stack
- **Python**
- **Ultralytics YOLOv8**
- **OpenCV**
- **NumPy**
- **Deep Learning–based Object Detection**

---

## ⚙️ How It Works
1. YOLOv8 detects pedestrians (COCO class ID: 0) in each frame.
2. The tracker assigns internal IDs to detected persons.
3. A person is assigned a custom ID only after appearing consistently across multiple frames.
4. The center point of each bounding box is tracked to generate motion trails.
5. Bounding boxes, IDs, and trajectories are visualized in real time.

---

## ▶️ How to Run
```bash
pip install ultralytics opencv-python numpy
