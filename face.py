import cv2
import face_recognition
import os
from fer import FER
import pyttsx3

# ✅ Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ✅ Load FER emotion detector
emotion_detector = FER()

# ✅ Load known face encodings
known_face_encodings = []
known_face_names = []

print("📁 Loading known faces from 'images/' folder...")
for filename in os.listdir("images"):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join("images", filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
            print(f"✅ Loaded {filename}")
        else:
            print(f"⚠ No face found in {filename}, skipping.")

# ✅ Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

# ✅ Memory for previous face state
previous_faces = {}  # {name: {'emotion': 'Happy', 'seen': True}}

print("📸 Starting face + emotion detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    current_faces = {}

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if matches:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # ✅ Face crop with padding
        padding = 20
        h, w, _ = frame.shape
        top_pad = max(0, top - padding)
        bottom_pad = min(h, bottom + padding)
        left_pad = max(0, left - padding)
        right_pad = min(w, right + padding)
        face_img = frame[top_pad:bottom_pad, left_pad:right_pad]

        # ✅ Emotion detection
        emotion = "N/A"
        if face_img.size != 0:
            try:
                face_img = cv2.resize(face_img, (128, 128))
                result = emotion_detector.detect_emotions(face_img)
                if result:
                    emotions = result[0]["emotions"]
                    emotion, confidence = max(emotions.items(), key=lambda x: x[1])
                    if confidence < 0.4:
                        emotion = "Unclear"
            except Exception as e:
                print(f"Emotion detection error: {e}")

        # ✅ Determine if we need to speak
        speak_update = False
        if name not in previous_faces:
            speak_update = True
        elif previous_faces[name]["emotion"] != emotion:
            speak_update = True
        elif not previous_faces[name]["seen"]:
            speak_update = True

        if speak_update:
            label = f"{name} looks {emotion}"
            print(f"🗣 {label}")
            engine.say(label)
            engine.runAndWait()

        # ✅ Mark as currently seen
        current_faces[name] = {'emotion': emotion, 'seen': True}

        # ✅ Draw results
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({emotion})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ✅ Update face tracking status
    for name in previous_faces:
        if name not in current_faces:
            previous_faces[name]["seen"] = False  # Face left the frame

    for name in current_faces:
        previous_faces[name] = current_faces[name]  # Update or add

    # ✅ Show frame
    cv2.imshow("Face Recognition + Emotion + Voice", frame)
    if cv2.waitKey(1) == ord('q'):
        print("🛑 Quitting.")
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
engine.stop()