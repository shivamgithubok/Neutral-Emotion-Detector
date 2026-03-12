import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, frame = cap.read()
    if not success:
        print("Camera not detected")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    # Draw face mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION
            )

    # Show window
    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()