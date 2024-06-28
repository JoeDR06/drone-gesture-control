import mediapipe as mp
import cv2
from mediapipe.framework.formats import landmark_pb2

# Import the necessary classes and modules from the Mediapipe library
base_options = mp.tasks.BaseOptions
gesture_recognizer = mp.tasks.vision.GestureRecognizer
gesture_recognizer_options = mp.tasks.vision.GestureRecognizerOptions
gesture_recognizer_result = mp.tasks.vision.GestureRecognizerResult
vision_running_mode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open the default camera
camera = cv2.VideoCapture(0)

# Set up the Gesture Recognizer options
options = gesture_recognizer_options(
    base_options = base_options(model_asset_path = "./models/gesture_recognizer.task"),
    running_mode = vision_running_mode.IMAGE,
    num_hands=10
)

# Create the Gesture Recognizer instance
with gesture_recognizer.create_from_options(options) as recognizer:
    while True:
        # Capture a frame from the camera
        sucess, frame = camera.read()

        # Convert the frame to a Mediapipe Image format
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)

        # Recognize the gestures in the frame
        result = recognizer.recognize(mp_image)

        # Check if there are any hand landmarks detected
        if len(result.hand_landmarks) == 0:
            # If no hands are detected, display the frame and check for user input
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) in [ord("q"), 27, 32]:
                break
            continue

        # Draw the hand landmarks on the frame
        for i in range(len(result.hand_landmarks)):
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in result.hand_landmarks[i]])

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Display the frame with the hand landmarks
        cv2.imshow("frame", frame)

        # Check for user input to exit the loop
        if cv2.waitKey(1) in [ord("q"), 27, 32]:
            break