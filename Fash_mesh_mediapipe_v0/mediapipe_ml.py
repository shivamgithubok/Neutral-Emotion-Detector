import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained emotion model
model = load_model("best_model.h5", compile=False)

emotion_labels = ["angry","disgust","fear","happy","sad","neutral","surprised"]

# MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Fix negative coordinates
            x = max(0, x)
            y = max(0, y)

            face = frame[y:y+height, x:x+width]

            if face.size == 0:
                continue

            # Preprocess for MobileNetV2
            face = cv2.resize(face, (224,224))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw box
            cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)

            cv2.putText(frame,
                        emotion,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()