from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/yolov11_soccer.pt')

# Load video
cap = cv2.VideoCapture('15sec_input_720p.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 on frame
    results = model(frame)

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Show frame (optional)
    cv2.imshow('Detections', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
