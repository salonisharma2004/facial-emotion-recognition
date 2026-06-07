
"""
====================================================================
  FACIAL EMOTION RECOGNITION SYSTEM
  Uses: OpenCV, DeepFace, Haar Cascade Classifier
  Detects: 7 emotions — happy, sad, angry, fear, surprise, disgust, neutral
====================================================================

SETUP — Install dependencies first:
    pip install opencv-python deepface tf-keras

HOW IT WORKS (overview):
  1. OpenCV captures video frames from your webcam
  2. Haar Cascade finds face regions in each frame (fast, CPU-friendly)
  3. DeepFace runs a pre-trained CNN on each detected face
  4. The top emotion + confidence bar are drawn on screen

====================================================================
"""

import cv2                     # OpenCV — camera capture, drawing, image ops
import numpy as np             # Numerical arrays (used for bar chart rendering)
from deepface import DeepFace  # Pre-trained emotion CNN wrapped in a simple API
import time                    # Used to throttle analysis (avoid overloading CPU)


# ── STEP 1: Load the Haar Cascade face detector ──────────────────────────────
#
# A Haar Cascade is a trained binary classifier that slides a small window
# across the image looking for "face-like" patterns (eyes above nose above mouth).
# It is fast but only detects frontal faces reliably.
#
# opencv comes with pre-trained XMLs for faces, eyes, smiles, etc.
# 'haarcascade_frontalface_default.xml' detects forward-facing faces.
#
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ── STEP 2: Define emotion metadata ──────────────────────────────────────────
#
# These are the 7 Universal Emotions defined by Paul Ekman (1972).
# DeepFace returns a dict with these exact keys.
# We assign a colour (BGR, not RGB!) to each emotion for the on-screen bar chart.
#
EMOTION_COLORS = {
    'happy':    (0, 220, 100),   # green
    'neutral':  (160, 160, 160), # gray
    'sad':      (220, 80,  40),  # blue-ish
    'angry':    (40,  40, 230),  # red
    'surprise': (0, 190, 255),   # yellow
    'fear':     (130, 60, 200),  # purple
    'disgust':  (50, 200, 200),  # teal
}

# Friendly display labels (capitalised for the UI)
EMOTION_LABELS = {k: k.capitalize() for k in EMOTION_COLORS}


# ── STEP 3: Open the webcam ───────────────────────────────────────────────────
#
# cv2.VideoCapture(0) opens the default camera (index 0).
# Change to 1 or 2 if you have multiple cameras and 0 is wrong.
#
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check that no other app is using it.")

# Optional: set resolution (lower = faster processing)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Webcam opened. Press 'q' to quit.")


# ── STEP 4: Per-face state (smoothing + throttle) ─────────────────────────────
#
# DeepFace is relatively slow (~200 ms per face). We only re-analyse a face
# every ANALYSIS_INTERVAL seconds and display the cached result in between.
#
# Smoothing: we keep a rolling average of the last N emotion score sets so
# the on-screen bar doesn't jump around on every frame.
#
ANALYSIS_INTERVAL  = 0.4   # seconds between DeepFace calls
SMOOTHING_FRAMES   = 5     # number of past results to average

last_analysis_time  = 0.0
smoothed_emotions   = {e: 0.0 for e in EMOTION_COLORS}
emotion_history     = []   # list of dicts; capped at SMOOTHING_FRAMES


# ── HELPER: Draw the emotion bar chart ───────────────────────────────────────
#
# Draws a small vertical panel of emotion bars to the right of each face rect.
#
def draw_emotion_bars(frame, x, y, w, scores: dict):
    """
    frame  : the BGR image array to draw onto
    x, y   : top-left corner of the face bounding box
    w      : width of the face bounding box (used to position the panel)
    scores : dict { emotion_name: float (0-100) }
    """
    panel_x    = x + w + 10       # Panel sits just to the right of the face box
    bar_height = 14
    bar_gap    = 6
    bar_max_w  = 130
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38

    for i, (emotion, color) in enumerate(EMOTION_COLORS.items()):
        score     = scores.get(emotion, 0.0)
        bar_w     = int(bar_max_w * score / 100.0)
        by        = y + i * (bar_height + bar_gap)

        # Background track
        cv2.rectangle(frame,
                      (panel_x, by),
                      (panel_x + bar_max_w, by + bar_height),
                      (50, 50, 50), -1)

        # Filled portion
        if bar_w > 0:
            cv2.rectangle(frame,
                          (panel_x, by),
                          (panel_x + bar_w, by + bar_height),
                          color, -1)

        # Label + percentage text
        label = f"{EMOTION_LABELS[emotion]}: {score:.0f}%"
        cv2.putText(frame, label,
                    (panel_x + bar_max_w + 6, by + bar_height - 3),
                    font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)


# ── STEP 5: Main loop ─────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()   # ret = success bool, frame = BGR numpy array

    if not ret:
        print("[WARN] Frame read failed — skipping.")
        continue

    # ── 5a. Convert to greyscale for face detection ───────────────────────────
    #
    # Haar Cascade works on greyscale; colour info is irrelevant for it.
    # Equalising the histogram makes detection more robust to dim lighting.
    #
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ── 5b. Detect faces ──────────────────────────────────────────────────────
    #
    # detectMultiScale parameters:
    #   scaleFactor  – how much to shrink the image at each scale (1.1 = 10% smaller each pass)
    #   minNeighbors – how many overlapping detections a candidate must have to be kept
    #                  (higher → fewer false positives, but may miss real faces)
    #   minSize      – ignore any detection smaller than this (filters tiny noise)
    #
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (60, 60)
    )

    now = time.time()

    # ── 5c. Analyse each face ─────────────────────────────────────────────────
    for (fx, fy, fw, fh) in faces:

        # Draw the face bounding box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 180), 2)

        # Only call DeepFace every ANALYSIS_INTERVAL seconds
        if now - last_analysis_time >= ANALYSIS_INTERVAL:
            last_analysis_time = now

            # Crop the face region from the original colour frame
            face_roi = frame[fy : fy + fh, fx : fx + fw]

            try:
                # ── DeepFace.analyze() ────────────────────────────────────────
                #
                # Under the hood, DeepFace resizes the crop to 48×48,
                # runs it through a small CNN trained on FER-2013
                # (35,887 labelled face images), and returns softmax probabilities
                # for the 7 emotion classes.
                #
                # actions=['emotion'] skips age/gender/race detection (faster).
                # enforce_detection=False lets it proceed even if its own
                # face detector is uncertain (we already found the face).
                #
                result = DeepFace.analyze(
                    face_roi,
                    actions           = ['emotion'],
                    enforce_detection = False,
                    silent            = True
                )

                # result is a list when multiple faces are found inside the ROI
                if isinstance(result, list):
                    result = result[0]

                raw_scores = result.get('emotion', {})   # { emotion: float (0-100) }

                # Normalise so scores sum to 100 (floating-point rounding may drift)
                total = sum(raw_scores.values()) or 1.0
                norm_scores = {k: (v / total) * 100 for k, v in raw_scores.items()}

                # Add to history and cap length
                emotion_history.append(norm_scores)
                if len(emotion_history) > SMOOTHING_FRAMES:
                    emotion_history.pop(0)

                # Average scores across history
                smoothed_emotions = {
                    e: np.mean([h.get(e, 0.0) for h in emotion_history])
                    for e in EMOTION_COLORS
                }

            except Exception as exc:
                # Analysis can fail on very small/blurry crops — silently skip
                pass

        # ── 5d. Display dominant emotion label ───────────────────────────────
        dominant = max(smoothed_emotions, key=smoothed_emotions.get)
        confidence = smoothed_emotions[dominant]
        color = EMOTION_COLORS[dominant]

        label_text = f"{EMOTION_LABELS[dominant]}  {confidence:.0f}%"

        # Semi-transparent filled rectangle behind the text for readability
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame,
                      (fx, fy - th - 12),
                      (fx + tw + 10, fy),
                      (0, 0, 0), -1)

        cv2.putText(frame, label_text,
                    (fx + 5, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # ── 5e. Draw emotion bar chart panel ─────────────────────────────────
        draw_emotion_bars(frame, fx, fy, fw, smoothed_emotions)

    # ── 5f. Overlay instructions ──────────────────────────────────────────────
    cv2.putText(frame, "Press 'q' to quit",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # ── 5g. Show frame ────────────────────────────────────────────────────────
    cv2.imshow("Facial Emotion Recognition", frame)

    # ── 5h. Key handling ──────────────────────────────────────────────────────
    #
    # waitKey(1) waits 1 ms for a keypress and returns the ASCII code.
    # 0xFF masks to the lower byte (needed on some 64-bit systems).
    #
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quitting.")
        break
    elif key == ord('s'):
        # Press 's' to save a screenshot
        filename = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved: {filename}")
    elif key == ord('r'):
        # Press 'r' to reset the smoothing history
        emotion_history.clear()
        smoothed_emotions = {e: 0.0 for e in EMOTION_COLORS}
        print("[INFO] Emotion history reset.")


# ── STEP 6: Release resources ─────────────────────────────────────────────────
#
# Always release the camera and destroy OpenCV windows on exit,
# otherwise the webcam stays "in use" and the process may hang.
#
cap.release()
cv2.destroyAllWindows()
print("[INFO] Resources released. Goodbye!")
