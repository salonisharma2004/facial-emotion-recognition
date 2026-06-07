# 🎭 Facial Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![DeepFace](https://img.shields.io/badge/DeepFace-latest-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

Real-time detection and classification of **7 human emotions** from a live webcam feed using **Haar Cascade** face detection and a **pre-trained CNN** via DeepFace.

---

## 📸 Demo

> Webcam opens → face is detected → emotion label + confidence bars appear in real time.

```
😊 Happy  72%   ████████████░░░░
😐 Neutral 15%  ███░░░░░░░░░░░░░
😢 Sad      5%  █░░░░░░░░░░░░░░░
😠 Angry    3%  ░░░░░░░░░░░░░░░░
😱 Surprise 3%  ░░░░░░░░░░░░░░░░
😨 Fear     1%  ░░░░░░░░░░░░░░░░
🤢 Disgust  1%  ░░░░░░░░░░░░░░░░
```

---

## ✨ Features

- 🎥 **Live webcam** — real-time processing frame by frame
- 🔍 **Haar Cascade** face detector — fast, CPU-friendly frontal face detection
- 🧠 **DeepFace CNN** — pre-trained on FER-2013 (35,887 labelled face images)
- 📊 **Emotion bar chart** — all 7 scores displayed alongside the face
- 🔄 **Smoothing** — averages last 5 results to prevent flickering
- ⚡ **Throttled analysis** — DeepFace runs every 0.4s to keep CPU usage low
- ⌨️ **Keyboard controls** — quit, screenshot, reset smoothing

---

## 🧠 Emotions Detected

Based on Paul Ekman's 7 Universal Emotions (1972):

| Emotion | Label |
|---------|-------|
| 😊 | Happy |
| 😢 | Sad |
| 😠 | Angry |
| 😨 | Fear |
| 😱 | Surprise |
| 🤢 | Disgust |
| 😐 | Neutral |

---

## 🗂️ Project Structure

```
facial-emotion-recognition/
│
├── emotion_detection.py     # Main script — webcam loop + detection + drawing
├── requirements.txt         # Python dependencies
└── README.md                # You are here
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| OpenCV (`cv2`) | Webcam capture, image drawing, Haar Cascade |
| DeepFace | Pre-trained emotion CNN (FER-2013) |
| NumPy | Score smoothing & array operations |
| Haar Cascade XML | Fast frontal face detection (built into OpenCV) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/salonisharma2004/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python emotion_detection.py
```

> **Note:** DeepFace will auto-download the emotion model (~80MB) on first run. Make sure you have an internet connection.

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save a screenshot |
| `R` | Reset smoothing history |

---

## 🔧 How It Works

```
Webcam Frame
     │
     ▼
Convert to Greyscale + Histogram Equalisation
     │
     ▼
Haar Cascade → detect face bounding boxes (x, y, w, h)
     │
     ▼
Crop face ROI → pass to DeepFace.analyze()  (every 0.4s)
     │
     ▼
CNN returns scores for 7 emotions  (softmax probabilities)
     │
     ▼
Average last 5 results (smoothing)
     │
     ▼
Draw bounding box + label + bar chart on frame
     │
     ▼
cv2.imshow() → display to screen
```

---

## 📦 Requirements

```
opencv-python
deepface
tf-keras
numpy
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not opening | Change `cv2.VideoCapture(0)` to `(1)` or `(2)` |
| Too many false face detections | Increase `minNeighbors` from `5` to `7` |
| Missing faces | Decrease `scaleFactor` from `1.1` to `1.05` |
| Slow performance | Lower resolution: set width to `640`, height to `480` |
| DeepFace model not downloading | Check internet connection on first run |

---

## 🌱 Future Improvements

- [ ] Add support for image/video file input (not just webcam)
- [ ] Export emotion logs to CSV
- [ ] Build a Flask web app for browser-based demo
- [ ] Train a custom CNN on a larger dataset
- [ ] Add multi-face tracking with unique IDs

---

## 👩‍💻 Author

**Saloni Sharma**
CS Student | Python | ML | Computer Vision
[GitHub](https://github.com/salonisharma2004)

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.
