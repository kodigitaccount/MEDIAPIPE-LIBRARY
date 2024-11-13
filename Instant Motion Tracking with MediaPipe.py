import cv2
import mediapipe as mp

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Set up video capture
    cap = cv2.VideoCapture(0)  # Change the index if you want to use a different camera

    # Set up MediaPipe Hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
        while cap.isOpened():
            # Read frame from camera
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB and process it with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the frame with hand landmarks
            cv2.imshow('Instant Motion Tracking', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
