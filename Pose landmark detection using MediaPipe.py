import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a PoseLandmarker object.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True)

# Step 2: Load the input image.
image = cv2.imread(r"C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\2. DATASCIENCE PROJECT\29. Mediapipe\img2.jpeg")

# Step 3: Convert the image to RGB format and process it.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = detector.process(image_rgb)

# Step 4: Process the detection result and visualize it.
annotated_image = image.copy()
mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

# Step 5: Display the annotated image.
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
