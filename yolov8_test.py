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

def process_frame(frame):
    # Perform object detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Skip if not a person or low confidence
            if class_id != 0 or confidence < CONFIDENCE_THRESHOLD:
                continue

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOUNDING_BOX_COLOR, 3)

            # Add text with confidence
            label = f"Person {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

            print(f"Detected: Person (Confidence: {confidence:.2f})")

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
        cv2.imshow('Human Detection', processed_frame)

        # Break the loop if 'q', 'ESC', or 'SPACE' is pressed
        if cv2.waitKey(1) in [ord('q'), 27, 32]:
            break

    # Release the capture and close windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()