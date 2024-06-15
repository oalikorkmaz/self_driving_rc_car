import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2, Preview


model = YOLO('best.pt')


picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def process_frame(frame):
    
    results = model(frame)
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes.xyxy.numpy()  
        confidences = result.boxes.conf.numpy()  
        class_ids = result.boxes.cls.numpy()  

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f'{model.names[int(class_id)]} {confidence:.2f}'
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame

while True:
    
    frame = picam2.capture_array()
    
    
    annotated_frame = process_frame(frame)
    
    
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
