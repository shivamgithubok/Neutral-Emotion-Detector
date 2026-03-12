import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load trained emotion model
print("Loading emotion model...")
model = load_model("best_model.h5", compile=False)
print("✓ Model loaded!")

# Emotion labels
emotion_labels = ["angry", "confused", "happy", "neutral", "sad"]

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("✓ Face detector loaded!")

# Color mapping for emotions
emotion_colors = {
    "angry": (0, 0, 255),      # Red
    "confused": (0, 165, 255),  # Orange
    "happy": (0, 255, 0),       # Green
    "neutral": (255, 255, 0),   # Cyan
    "sad": (255, 0, 0)          # Blue
}

# Open webcam
print("\n▶ Starting webcam... (Press ESC to quit)\n")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Error: Cannot open webcam")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
face_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame_count += 1
        h, w, c = frame.shape

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # Process each detected face
        for (x, y, width, height) in faces:
            face_count += 1

            # Extract face region
            face = frame[y:y+height, x:x+width]

            if face.size == 0:
                continue

            # Preprocess for model (224x224, normalized)
            face_resized = cv2.resize(face, (224, 224))
            face_normalized = face_resized.astype("float32") / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)

            # Get emotion prediction
            prediction = model.predict(face_batch, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = emotion_labels[emotion_idx]
            confidence = float(prediction[0][emotion_idx]) * 100

            # Get color for emotion
            color = emotion_colors.get(emotion, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

            # Prepare text
            label_text = f"{emotion.upper()} ({confidence:.1f}%)"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

            # Draw background for text
            cv2.rectangle(frame,
                         (x, y - 35),
                         (x + text_size[0] + 10, y),
                         (0, 0, 0),
                         -1)
            cv2.rectangle(frame,
                         (x, y - 35),
                         (x + text_size[0] + 10, y),
                         color,
                         2)

            # Draw emotion label
            cv2.putText(frame,
                       label_text,
                       (x + 5, y - 10),
                       font,
                       font_scale,
                       color,
                       thickness)

            # Draw confidence bar below face
            bar_height = 8
            bar_width = int(confidence / 100 * width)
            cv2.rectangle(frame,
                         (x, y + height),
                         (x + bar_width, y + height + bar_height),
                         color,
                         -1)
            cv2.rectangle(frame,
                         (x, y + height),
                         (x + width, y + height + bar_height),
                         (80, 80, 80),
                         1)

            # Draw all emotion probabilities
            y_text = y + height + 30
            for i, (label, prob) in enumerate(zip(emotion_labels, prediction[0])):
                prob_percent = prob * 100
                bar_len = int(prob * 80)
                
                # Emotion name
                cv2.putText(frame,
                           f"{label}:",
                           (x, y_text),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4,
                           (200, 200, 200),
                           1)
                
                # Probability bar
                cv2.rectangle(frame,
                             (x + 65, y_text - 8),
                             (x + 65 + bar_len, y_text + 4),
                             emotion_colors.get(label, (100, 100, 100)),
                             -1)
                
                # Percentage
                cv2.putText(frame,
                           f"{prob_percent:.0f}%",
                           (x + 150, y_text),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4,
                           (200, 200, 200),
                           1)
                
                y_text += 18

        # Display stats
        stats_text = f"Faces: {len(faces)} | Frame: {frame_count} | Total detected: {face_count}"
        cv2.putText(frame,
                   stats_text,
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   (0, 255, 0),
                   2)

        # Show frame
        cv2.imshow("Emotion Detection - Press ESC to quit", frame)

        # Exit on ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

except Exception as e:
    print(f"\n✗ Error during processing: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Application closed")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total faces detected: {face_count}")