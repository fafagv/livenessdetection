import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def liveness_detection(frame):
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        # Apply liveness detection algorithm (e.g., texture analysis, motion analysis, etc.)
        # Replace the following lines with your chosen liveness detection approach

        # Example: Simple texture analysis using Laplacian variance
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        if laplacian_var < 100:  # Adjust this threshold based on your experiment
            cv2.putText(frame, "Real", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Fake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = liveness_detection(frame)

    cv2.imshow('Liveness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
