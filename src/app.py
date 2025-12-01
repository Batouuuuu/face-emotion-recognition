import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Charger le modèle
model = load_model("./data/cnn_emotion_detector.h5")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Lissage des prédictions
N_FRAMES = 10
pred_queue = deque(maxlen=N_FRAMES)

# Ouvrir la webcam
stream = cv2.VideoCapture(0)
if not stream.isOpened():
    raise RuntimeError("⚠️ Impossible d'ouvrir la webcam")

while True:
    ret, frame = stream.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))

    for (x, y, w, h) in faces:
    # Agrandir légèrement le rectangle : 10% en haut/bas et 10% à gauche/droite
        x1 = max(0, x - int(0.1 * w))
        y1 = max(0, y - int(0.1 * h))
        x2 = min(gray.shape[1], x + w + int(0.1 * w))
        y2 = min(gray.shape[0], y + h + int(0.2 * h))  # plus d’espace en bas pour la bouche

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        roi = gray[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_resized = cv2.equalizeHist(roi_resized)
        roi_norm = roi_resized / 255.0

        input_img = np.expand_dims(roi_norm, axis=(0,-1))
        pred = model.predict(input_img, verbose=0)[0]

        pred_queue.append(pred)
        avg_pred = np.mean(pred_queue, axis=0)

        emotion_index = np.argmax(avg_pred)
        emotion = emotions[emotion_index]
        confidence = avg_pred[emotion_index]

        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.release()
cv2.destroyAllWindows()
