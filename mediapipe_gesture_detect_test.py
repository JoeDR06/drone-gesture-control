import mediapipe as mp
import cv2
from mediapipe.framework.formats import landmark_pb2

base_options = mp.tasks.BaseOptions
gesture_recognizer = mp.tasks.vision.GestureRecognizer
gesture_recognizer_options = mp.tasks.vision.GestureRecognizerOptions
gesture_recognizer_result = mp.tasks.vision.GestureRecognizerResult
vision_running_mode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
camera = cv2.VideoCapture(0)
    
    
options = gesture_recognizer_options(
    base_options = base_options(model_asset_path = "./models/gesture_recognizer.task"),
    running_mode = vision_running_mode.IMAGE,
    num_hands=10
)

with gesture_recognizer.create_from_options(options) as recognizer:
    while True:
        sucess, frame = camera.read()
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
        result = recognizer.recognize(mp_image)
        print(len(result.hand_landmarks))
        if len(result.hand_landmarks) == 0:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) in [ord("q"), 27, 32]:
                break
            continue
        
        
        for i in range(len(result.hand_landmarks)):
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()        
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in result.hand_landmarks[i]])
        
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) in [ord("q"), 27, 32]:
            break