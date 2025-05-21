import cv2
from ultralytics import YOLO
import logging

# Disable YOLO logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8x.pt")  # You can use yolov8s.pt, yolov8m.pt, etc. depending on your model size

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the camera feed...")

while True:
    # Capture a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection using the model
    results = model(frame)

    # Access the first result and render it
    rendered_frame = results[0].plot()  # Use plot() to get the rendered image

    # Display the frame
    cv2.imshow("YOLOv8 Object Recognition", rendered_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()