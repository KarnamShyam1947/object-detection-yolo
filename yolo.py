from ultralytics import YOLO

model = YOLO('best_50.pt')
model.predict(source=0, show=True)
