import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("best_model.h5", compile=False)

# Correct labels from training
emotion_labels = ["angry","confused","happy","neutral","sad"]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to model input
    img = cv2.resize(frame, (224,224))

    # Normalize
    img = img.astype("float32") / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    emotion = emotion_labels[np.argmax(prediction)]

    # Display emotion
    cv2.putText(frame,
                emotion,
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()