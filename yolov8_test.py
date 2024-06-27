from ultralytics import YOLO
import cv2
import math

# Constants
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
MODEL_PATH = "yolo-Weights/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
BOUNDING_BOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, WEBCAM_WIDTH)
webcam.set(4, WEBCAM_HEIGHT)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Define object classes
class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def process_frame(frame):
    # Perform object detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Skip low confidence detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Get class name
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOUNDING_BOX_COLOR, 3)

            # Add text with class name and confidence
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

    return frame

def main():
    while True:
        # Capture frame-by-frame
        success, frame = webcam.read()
        if not success:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Object Detection', processed_frame)

        # Break the loop if 'q', 'ESC', or 'SPACE' is pressed
        if cv2.waitKey(1) in [ord('q'), 27, 32]:
            break

    # Release the capture and close windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()