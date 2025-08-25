object detection "import cv2
from ultralytics import YOLO
import pyttsx3

# ✅ Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking speed if needed

# ✅ Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for speed

# ✅ Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

# ✅ Track previous detections
previous_detected_names = set()

print("📸 Starting object detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # ✅ Run YOLOv8 detection
    results = model(frame, verbose=False)[0]

    # ✅ Get detected object names
    current_detected_names = set([model.names[int(cls)] for cls in results.boxes.cls])

    # ✅ Speak only if objects changed
    if current_detected_names != previous_detected_names:
        print(f"📤 Detected: {current_detected_names}")
        labels_to_speak = ", ".join(current_detected_names)
        engine.say(f"Detected: {labels_to_speak}")
        engine.runAndWait()
        previous_detected_names = current_detected_names

    # ✅ Draw boxes and labels
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ✅ Show frame
    cv2.imshow("YOLOv8 Detection with Voice", frame)
    if cv2.waitKey(1) == ord('q'):
        print("🛑 Quitting.")
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
engine.stop()
"