# ----> VideoCapture(0) tries to capture from the webcam
# ----> cap = cv2.VideoCapture(0)

# ---> 3D OBJECT DETECTION FROM VIDEO

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(r"C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\2. DATASCIENCE PROJECT\29. Mediapipe\pexels_videos.mp4")

objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=5,
                                   min_detection_confidence=0.4,
                                   min_tracking_confidence=0.70,
                                   model_name='Cup')

# Read video stream and feed into the model
while cap.isOpened():
    success, image = cap.read()

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(image,
                                      detected_object.landmarks_2d,
                                      mp_objectron.BOX_CONNECTIONS)

            mp_drawing.draw_axis(image,
                                 detected_object.rotation,
                                 detected_object.translation)

    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()