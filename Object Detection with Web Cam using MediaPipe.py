# IMPORT LIBRARY

import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection() as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        while not success:
            success, image = cap.read()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))

        # Exit on esc
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()