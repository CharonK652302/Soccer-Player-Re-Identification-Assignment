import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO('models/yolov11_soccer.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture('15sec_input_720p.mp4')
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 detection
    results = model(frame)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(w), int(h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
