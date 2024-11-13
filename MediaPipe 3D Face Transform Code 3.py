import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe face mesh and drawing utilities
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

def transform_3d_face(image, landmarks, replacement_face):
    # Perform 3D face transformation
    transformed_image = image.copy()

    # Get the bounding box coordinates of the face region
    xmin, ymin, xmax, ymax = get_face_bbox(landmarks, image.shape[:2])

    # Resize the replacement face image to match the face region
    resized_replacement_face = cv2.resize(replacement_face, (xmax - xmin, ymax - ymin))

    # Create a mask from the grayscale image of the replacement face
    mask = cv2.cvtColor(resized_replacement_face, cv2.COLOR_BGR2GRAY) / 255.0

    # Remove the alpha channel from the replacement face image
    replacement_face_rgb = resized_replacement_face[:, :, :3]

    # Calculate the region of interest for the replacement face
    roi = transformed_image[ymin:ymax, xmin:xmax]

    # Apply the replacement face to the region of interest
    roi = roi * (1 - mask[:, :, np.newaxis]) + replacement_face_rgb * mask[:, :, np.newaxis]

    # Place the modified region of interest back into the image
    transformed_image[ymin:ymax, xmin:xmax] = roi

    return transformed_image

def get_face_bbox(landmarks, image_shape):
    x_coordinates = [landmark[0] for landmark in landmarks]
    y_coordinates = [landmark[1] for landmark in landmarks]

    xmin = int(min(x_coordinates) * image_shape[1])
    ymin = int(min(y_coordinates) * image_shape[0])
    xmax = int(max(x_coordinates) * image_shape[1])
    ymax = int(max(y_coordinates) * image_shape[0])

    return xmin, ymin, xmax, ymax

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the replacement face image from webcam
ret, replacement_face = cap.read()

# Check if the replacement face image is captured successfully
if not ret:
    print("Failed to capture replacement face image from webcam.")
    exit()

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
            # Convert face landmarks to a list of tuples
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

            # Perform 3D face transformation
            transformed_image = transform_3d_face(image, landmarks, replacement_face)

            # Draw the face mesh on the image
            mp_drawing.draw_landmarks(
                transformed_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Create a composite image with the sketch on the left and the original frame on the right
    composite_image = np.hstack((replacement_face, transformed_image))

    # Show the composite image
    cv2.imshow('MediaPipe 3D Face Transform', composite_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
