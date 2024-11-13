import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe face mesh and drawing utilities
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

# Set the transformation matrix (example: scaling)
transformation_matrix = np.array([[1.5, 0, 0],
                                  [0, 1.5, 0],
                                  [0, 0, 1]])

def transform_3d_face(image, landmarks):
    # Perform 3D face transformation
    transformed_landmarks = np.matmul(landmarks, transformation_matrix.T)
    transformed_image = image.copy()

    for i in range(transformed_landmarks.shape[0]):
        x, y, _ = transformed_landmarks[i]

        x = int(x * image.shape[1])
        y = int(y * image.shape[0])

        transformed_landmarks[i] = [x, y, _]

        cv2.circle(transformed_image, (x, y), 1, (255, 0, 0), -1)

    return transformed_image

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

    # Process the image with MediaPipe face mesh
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert face landmarks to a NumPy array
            landmarks = np.zeros((468, 3), dtype=np.float32)
            for i, landmark in enumerate(face_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]

            # Perform 3D face transformation
            transformed_image = transform_3d_face(image, landmarks)

            # Draw the face mesh on the image
            mp_drawing.draw_landmarks(
                transformed_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Show the transformed image
    cv2.imshow('MediaPipe 3D Face Transform', transformed_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        breakQ

cap.release()
cv2.destroyAllWindows()
