import cv2
import os
import numpy as np
from ultralytics import YOLO
import gradio as gr

# CONFIG
video_paths = {
    "cam_139": "D:/peoplenet/data/videos/cam_139.mp4",
    "cam_140": "D:/peoplenet/data/videos/cam_140.mp4",
    "cam_142": "D:/peoplenet/data/videos/cam_142.mp4",
    "cam_52":  "D:/peoplenet/data/videos/cam_52.mp4"
}
output_dir = "D:/peoplenet/output"
save_every_sec = 25
CONF_THRESHOLD = 0.72

# PREP
os.makedirs(output_dir, exist_ok=True)
for cam in video_paths:
    os.makedirs(os.path.join(output_dir, cam), exist_ok=True)

model = YOLO("yolov8n.pt").to("cuda")
caps = {cam: cv2.VideoCapture(path) for cam, path in video_paths.items()}
fps = {cam: caps[cam].get(cv2.CAP_PROP_FPS) or 25 for cam in caps}
save_every_frame = {cam: int(fps[cam] * save_every_sec) for cam in fps}
frame_counts = {cam: 0 for cam in caps}

def process_frame():
    frames_out = []
    for cam, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frames_out.append(frame)
            continue

        frame_counts[cam] += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, verbose=False)[0]

        # Person only
        mask = (results.boxes.cls == 0) & (results.boxes.conf > CONF_THRESHOLD)
        people_boxes = results.boxes[mask]

        # Draw boxes
        for box in people_boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            conf = box.conf.cpu().numpy()[0]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Save frame if needed
        if frame_counts[cam] % save_every_frame[cam] == 0 and len(people_boxes) > 0:
            save_path = os.path.join(output_dir, cam, f"frame_{frame_counts[cam]:06d}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"✅ {cam}: Saved {save_path}")

        frames_out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 2x2 grid
    top = np.hstack((frames_out[0], frames_out[1]))
    bottom = np.hstack((frames_out[2], frames_out[3]))
    grid = np.vstack((top, bottom))
    grid = cv2.resize(grid, (1280, 720))
    return grid

def video_stream():
    while True:
        yield process_frame()

if __name__ == "__main__":
    gr.Interface(
        fn=video_stream,
        inputs=None,
        outputs=gr.Image(type="numpy", streaming=True),
        title="Live People Detection (2x2 Feed)",
        description="Detects people in 4 camera feeds with confidence > 0.72"
    ).launch(share=True)
