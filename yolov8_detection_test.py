from ultralytics import YOLO
import cv2
import math 

# Constants
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
MODEL_PATH = "yolo-Weights/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.8
BOUNDING_BOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2

# start webcam
cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# model
model = YOLO("models/yolov8n.pt")



while True:
    success, img = cap.read()
    results = model(img, stream=True)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), BOUNDING_BOX_COLOR, 3)

            # Add text with confidence
            label = f"Person {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

            print(f"Detected: Person (Confidence: {confidence:.2f})")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) in [ord('q'), 27, 32]:
        break

cap.release()
cv2.destroyAllWindows()