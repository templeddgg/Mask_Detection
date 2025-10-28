# train_model.py (แก้ใหม่)
from ultralytics import YOLO

# ใช้ yolov8s.pt → แม่นยำกว่า
model = YOLO('yolov8s.pt')

model.train(
    data='yolov8_dataset/data.yaml',
    epochs=20,
    imgsz=640,
    batch=16,
    name='mask_detection_v2',
    patience=5,
    augment=True  # เพิ่ม data augmentation
)