import torch
import cv2

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # 'yolov5x' is a larger model with better accuracy
model = model.to('cuda')  # Move the model to GPU if available

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

    # Convert the frame from BGR to RGB (YOLOv5 expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection using GPU
    results = model(frame_rgb)

    # Render the results on the frame (draw bounding boxes and labels)
    frame = results.render()[0]  # Rendered frame with detections

    # Convert the frame back to BGR for OpenCV to display
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow("YOLOv5 Object Recognition", frame_bgr)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()