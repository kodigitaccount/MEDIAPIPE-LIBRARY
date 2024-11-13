import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Check if landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Iterate through landmarks of each hand
            for landmark in hand_landmarks.landmark:
                # Get pixel coordinates of each landmark
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the frame with landmarks
    cv2.imshow('Hand Landmarks', frame)

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
