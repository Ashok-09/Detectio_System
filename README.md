# Detection_System
### ****
#  Blackbox Analytics – Intelligent Wearable Safety Device

##  Overview
**Blackbox Analytics** is an AI-powered wearable device designed to enhance **safety, independence, and well-being** for elderly and physically challenged individuals.  
It uses **Computer Vision** and **Artificial Intelligence** to detect falls, recognize emotions, and identify environmental hazards **in real time** — all processed **locally** without relying on cloud servers or internet connectivity.

---

##  Features
- **Face & Emotion Recognition**  
  Detects known individuals and monitors emotional states (stress, fatigue, distress).  
- **Object & Hazard Detection (YOLOv8)**  
  Recognizes obstacles like stairs, furniture, and vehicles with real-time alerts.  
- **Human Motion & Fall Detection (Mediapipe)**  
  Differentiates between walking, standing, and falling.  
- **On-Device Processing**  
  Runs entirely on **Raspberry Pi 4** or **Jetson Nano** for privacy and reliability.  
- **Multi-Modal Alerts**  
  Provides immediate **voice, buzzer, or vibration feedback**.  
- **Offline Event Logging**  
  Stores events locally for review without cloud dependency.  

---

##  System Architecture
```

\[Camera] → \[AI Inference Modules]
├─ Face & Emotion Recognition
├─ Object Detection (YOLOv8)
└─ Fall/Walking Detection (Mediapipe)
↓
\[Processing Engine] → \[Voice + Vibration Alerts] → \[Optional Local Logs]

```

---

##  Tech Stack
- **Hardware**: Raspberry Pi 4 / Jetson Nano, Camera Module, Buzzer/Vibration Motor, Speaker  
- **Programming Language**: Python 3  
- **Libraries/Frameworks**:  
  - OpenCV (image/video processing)  
  - face_recognition (face detection & recognition)  
  - FER (facial emotion recognition)  
  - YOLOv8 (object & hazard detection)  
  - Mediapipe (pose estimation & fall detection)  
  - pyttsx3 (offline text-to-speech)  

---

##  Project Structure
```

├── face\_emotion\_recognition.py   # Face detection + emotion recognition
├── object\_detection\_yolo.py      # YOLOv8-based object detection
├── fall\_detection\_mediapipe.py   # Fall & walking detection
├── alerts.py                     # Manages voice + haptic feedback
├── main.py                       # Orchestrates modules and alerts
├── requirements.txt              # Dependencies
├── images/                       # Known faces dataset
├── logs/                         # Local event logs



##  Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**

   ```bash
   python main.py
   ```


##  Performance Highlights

* **Fall Detection Accuracy**: 92%
* **Walking Pattern Accuracy**: 87%
* **Object Detection Speed**: 12–15 FPS on Raspberry Pi 4
* **Alert Latency**: \~1.4 seconds


##  Future Improvements

* Add **infrared/thermal sensors** for low-light performance
* Lightweight **offline dashboard** for reviewing logs
* **Power-efficient hardware** for longer wearable use
* Adaptive learning for **personalized alerts**
* Optional **secure cloud sync** for caregivers


##  License

This project is licensed under the **MIT License** – feel free to use, modify, and share.



##  Contributors

   Vishal Kavali ( Developer & Researcher)
   Pallapu Ashok ( Developer & Researcher)






