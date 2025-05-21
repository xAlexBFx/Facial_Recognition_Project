import cv2
import os
from deepface import DeepFace
import asyncio
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms as T
# Suppress DeepFace's internal logging (this redirects logs to null)

# Load models
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Set model to use FP16 precision if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if torch.cuda.is_available():
    model = model.half()

# Define transformations for images (downscaling)
preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Known faces folder path
known_faces_folder = 'known_faces'

async def recognize_faces(frame):
    try:
        recognized_names = []  # Store names for all detected faces

        # Load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Perform face recognition for each detected face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]  # Crop face region

            # Run DeepFace on the cropped face
            result = DeepFace.find(face_roi, db_path=known_faces_folder, enforce_detection=False, silent=True)

            if result and len(result[0]) > 0:  # Check if a match exists
                recognized_name = result[0]["identity"].iloc[0]
                recognized_name = os.path.splitext(os.path.basename(recognized_name))[0]
            else:
                recognized_name = "Unknown"

            recognized_names.append((recognized_name, (x, y, w, h)))

        # Draw rectangles & put names for each detected face
        for name, (x, y, w, h) in recognized_names:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return recognized_names, frame  # Return names & updated frame

    except Exception as e:
        print(f"[ERROR] Face recognition failed: {e}")
        return [("Unknown", (0, 0, 0, 0))], frame


async def generate_caption(frame):
    try:
        img = Image.fromarray(frame)

        # Preprocess image and run through the model
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        img_tensor = img_tensor.half() if torch.cuda.is_available() else img_tensor

        inputs = processor(img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error in caption generation"

# Analyze emotion using DeepFace (running it asynchronously as well)
async def analyze_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
        if(confidence < 90):
            return 'Not sure'
        return emotion
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return "Error in emotion analysis"

# Asynchronous frame processing
async def process_frame(frame):
    # Run the tasks concurrently
    caption_task = asyncio.create_task(generate_caption(frame))
    emotion_task = asyncio.create_task(analyze_emotion(frame))
    face_task = asyncio.create_task(recognize_faces(frame))

    # Unpack the results from the tasks
    caption, emotion, (recognized_name, frame_with_face) = await asyncio.gather(caption_task, emotion_task, face_task)

    # Print the output results for caption, emotion, and face recognition
    print(f"Recognized Name: {recognized_name}")
    print(f"Caption: {caption}")
    print(f"Emotion: {emotion}")

    return frame_with_face

# Real-time video capture loop (OpenCV)
async def process_real_time_frames():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # If the frame is not read properly, stop the loop

        # Process the frame asynchronously
        frame_with_face = await process_frame(frame)

        # Display the frame with the recognized face and name
        cv2.imshow("Frame", frame_with_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the real-time frame processing loop
asyncio.run(process_real_time_frames())