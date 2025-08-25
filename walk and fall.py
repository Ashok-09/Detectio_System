walk and fall detection "import cv2
import mediapipe as mp
import pyttsx3

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture (default webcam)
cap = cv2.VideoCapture(0)

# Store previous detection and positions
prev_detection = None
prev_shoulder = None
prev_hip = None

# Fall detection threshold
fall_threshold = 0.2

def speak(text):
    engine.say(text)
    engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detection_result = "No Detection"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract key landmarks
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        vertical_distance = abs(left_shoulder.y - left_hip.y)

        if prev_shoulder and prev_hip:
            shoulder_move = abs(left_shoulder.x - prev_shoulder[0]) + abs(right_shoulder.x - prev_shoulder[1])
            hip_move = abs(left_hip.x - prev_hip[0]) + abs(right_hip.x - prev_hip[1])

            if shoulder_move + hip_move > 0.1:
                detection_result = "Walking"

            if vertical_distance < fall_threshold:
                detection_result = "Fall Detected"

        # Update previous keypoints
        prev_shoulder = (left_shoulder.x, right_shoulder.x)
        prev_hip = (left_hip.x, right_hip.x)

    # Voice output only when detection changes to Walking or Fall Detected
    if detection_result != prev_detection and detection_result in ["Walking", "Fall Detected"]:
        print(f"🔊 {detection_result}")
        speak(detection_result)

    prev_detection = detection_result

    # Display the result
    if detection_result != "No Detection":
        cv2.putText(image, f'Detection: {detection_result}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Walk and Fall Detection (Local)', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
"