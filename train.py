from ultralytics import YOLO

model = YOLO('weights/yolov8s.pt')

model.train(
    data='data.yaml',
    epochs=20,
    optimizer='Adam',
    lr0=1e-3
)