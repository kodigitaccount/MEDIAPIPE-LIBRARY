import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide the path to a video file

# Initialize BlazePose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        # Read frame from video capture
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with BlazePose
        results = pose.process(image)

        if results.pose_landmarks:
            # Render the landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the frame
        cv2.imshow('Real-time Body Pose Tracking with BlazePose', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
