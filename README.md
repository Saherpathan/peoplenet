# PeopleNet: Multi-Camera People Detection

🚀 **PeopleNet** is a Python-based application for detecting people in multiple video streams using YOLOv8 and displaying the results in a live 2x2 grid via Gradio.

---

## 📌 Features

✅ Detects **people only** (YOLO class `0`) in 4 video feeds  
✅ Saves frames with detections every **25 seconds** per camera  
✅ Confidence threshold > **0.72**  
✅ Outputs a live 2x2 grid via Gradio (local + public shareable link)

---

## 🖥️ Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for optimal YOLOv8 speed)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Gradio
- OpenCV
- NumPy

### Install dependencies:
```bash
pip install ultralytics gradio opencv-python numpy
```
📂 Directory Structure

```
D:/peoplenet/
├── app/
│   ├── main.py # Main detection + Gradio app script
│   ├── test.py
│   ├── gradio_app.py         
├── data/
│   └── videos/
│       ├── cam_139.mp4
│       ├── cam_140.mp4
│       ├── cam_142.mp4
│       └── cam_52.mp4
└── output/
│   ├── cam_139/
│   ├── cam_140/
│   ├── cam_142/
│   └── cam_52/
├── README.md
└── requirements.txt
```

 How to Run
## ⚡ How to Run
Run this in your terminal:
```
python main.py
```
You will see:

Running on local URL:  http://127.0.0.1:7860

### 📝 How It Works
Loads YOLOv8 model (yolov8n.pt)

Opens 4 video streams

Detects only people (class 0) with confidence > 0.72

Displays live 2x2 grid using Gradio

Saves annotated frames every 25 seconds per camera (if person detected)

### ⚙️ Configurable Parameters
Parameter	Default	Description
save_every_sec	25	Save annotated frames every N seconds
CONF_THRESHOLD	0.72	Confidence threshold for person detection

### 📦 Output
Detected frames are saved as:

```bash
D:/peoplenet/output/{camera_name}/frame_000025.jpg
```
One subdirectory per camera.

