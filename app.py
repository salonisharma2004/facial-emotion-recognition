"""
Flask Web App for Facial Emotion Recognition
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import base64

app = Flask(__name__)

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTION_COLORS_BGR = {
    'happy':    (0, 220, 100),
    'neutral':  (160, 160, 160),
    'sad':      (220, 80,  40),
    'angry':    (40,  40, 230),
    'surprise': (0, 190, 255),
    'fear':     (130, 60, 200),
    'disgust':  (50, 200, 200),
}

EMOTION_COLORS_HEX = {
    'happy':    '#64dc64',
    'neutral':  '#a0a0a0',
    'sad':      '#e05028',
    'angry':    '#e02828',
    'surprise': '#ffbe00',
    'fear':     '#823cc8',
    'disgust':  '#32c8c8',
}

EMOTION_EMOJIS = {
    'happy': '😊', 'neutral': '😐', 'sad': '😢',
    'angry': '😠', 'surprise': '😱', 'fear': '😨', 'disgust': '🤢'
}

# ── Global state ──────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

camera = None
camera_lock = threading.Lock()

ANALYSIS_INTERVAL = 0.4
SMOOTHING_FRAMES  = 5

last_analysis_time = 0.0
smoothed_emotions  = {e: 0.0 for e in EMOTION_COLORS_BGR}
emotion_history    = []
current_stats      = {
    'dominant': 'neutral',
    'confidence': 0.0,
    'emotions': {e: 0.0 for e in EMOTION_COLORS_BGR},
    'face_count': 0,
}
stats_lock = threading.Lock()


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def draw_emotion_bars(frame, x, y, w, scores):
    panel_x    = x + w + 10
    bar_height = 14
    bar_gap    = 5
    bar_max_w  = 120
    font       = cv2.FONT_HERSHEY_SIMPLEX

    for i, (emotion, color) in enumerate(EMOTION_COLORS_BGR.items()):
        score = scores.get(emotion, 0.0)
        bar_w = int(bar_max_w * score / 100.0)
        by    = y + i * (bar_height + bar_gap)

        cv2.rectangle(frame, (panel_x, by), (panel_x + bar_max_w, by + bar_height), (30, 30, 30), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (panel_x, by), (panel_x + bar_w, by + bar_height), color, -1)
        label = f"{emotion.capitalize()}: {score:.0f}%"
        cv2.putText(frame, label, (panel_x + bar_max_w + 6, by + bar_height - 3),
                    font, 0.36, (200, 200, 200), 1, cv2.LINE_AA)


def generate_frames():
    global last_analysis_time, smoothed_emotions, emotion_history

    cap = get_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        now = time.time()

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 180), 2)

            if now - last_analysis_time >= ANALYSIS_INTERVAL:
                last_analysis_time = now
                face_roi = frame[fy:fy + fh, fx:fx + fw]

                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'],
                                              enforce_detection=False, silent=True)
                    if isinstance(result, list):
                        result = result[0]

                    raw_scores = result.get('emotion', {})
                    total = sum(raw_scores.values()) or 1.0
                    norm_scores = {k: (v / total) * 100 for k, v in raw_scores.items()}

                    emotion_history.append(norm_scores)
                    if len(emotion_history) > SMOOTHING_FRAMES:
                        emotion_history.pop(0)

                    smoothed_emotions = {
                        e: np.mean([h.get(e, 0.0) for h in emotion_history])
                        for e in EMOTION_COLORS_BGR
                    }
                except:
                    pass

            dominant   = max(smoothed_emotions, key=smoothed_emotions.get)
            confidence = smoothed_emotions[dominant]
            color      = EMOTION_COLORS_BGR[dominant]

            label_text = f"{dominant.capitalize()}  {confidence:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (fx, fy - th - 12), (fx + tw + 10, fy), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (fx + 5, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            draw_emotion_bars(frame, fx, fy, fw, smoothed_emotions)

        with stats_lock:
            current_stats['dominant']   = max(smoothed_emotions, key=smoothed_emotions.get)
            current_stats['confidence'] = smoothed_emotions[current_stats['dominant']]
            current_stats['emotions']   = dict(smoothed_emotions)
            current_stats['face_count'] = len(faces)

        cv2.putText(frame, "Facial Emotion Recognition | Press Q to quit",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        ret2, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html',
                           emotion_colors=EMOTION_COLORS_HEX,
                           emotion_emojis=EMOTION_EMOJIS)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    with stats_lock:
        data = {
            'dominant':   current_stats['dominant'],
            'confidence': round(current_stats['confidence'], 1),
            'emotions':   {k: round(v, 1) for k, v in current_stats['emotions'].items()},
            'face_count': current_stats['face_count'],
            'emoji':      EMOTION_EMOJIS.get(current_stats['dominant'], '😐'),
            'color':      EMOTION_COLORS_HEX.get(current_stats['dominant'], '#a0a0a0'),
        }
    return jsonify(data)


if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    print("[INFO] Open your browser at: http://localhost:5000")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
