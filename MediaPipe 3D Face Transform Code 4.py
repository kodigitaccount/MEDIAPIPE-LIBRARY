import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe pose and drawing utilities
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

# Set the range of colors to replace (in BGR format)
lower_color = np.array([0, 0, 0])  # lower range of color to replace
upper_color = np.array([50, 50, 50])  # upper range of color to replace

# Set the new color to replace the existing color (in BGR format)
new_color = np.array([0, 0, 255])  # new color (red)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe pose detection
    results = pose.process(image_rgb)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, drawing_spec, drawing_spec)

        # Get the coordinates of the left and right wrists
        left_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image.shape[1]),
                      int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image.shape[0]))
        right_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image.shape[1]),
                       int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image.shape[0]))

        # Create a mask for the range of colors to replace within the region of clothing
        mask = cv2.inRange(image, lower_color, upper_color)

        # Apply the new color to the mask
        colored_mask = cv2.bitwise_and(image, image, mask=mask)
        colored_mask[mask > 0] = new_color

        # Combine the original image and the colored mask
        result = cv2.bitwise_or(image, colored_mask)

        # Show the modified image
        cv2.imshow('Change Cloth Color', result)

    else:
        # Show the original image if no pose landmarks are detected
        cv2.imshow('Change Cloth Color', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
