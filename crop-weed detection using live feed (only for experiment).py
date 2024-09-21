import cv2
import numpy as np

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    return normalized_image

from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = 'crop_weed_detection_model.h5'  # Replace with your model path
model = load_model(model_path)

# Function to predict crop or weed
def predict_crop_weed(image):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    prediction = model.predict(preprocessed_image)
    confidence = prediction[0][0]
    label = 'Crop' if confidence > 0.5 else 'Weed'
    return label, confidence

# Function to capture and process live camera feed
def detect_crop_weed_live():
    cap = cv2.VideoCapture(0)  # Open default camera (index 0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break
        
        # Preprocess the frame and make prediction
        label, confidence = predict_crop_weed(frame)
        
        # Display result on the frame
        if label == 'Crop':
            color = (0, 255, 0)  # Green for Crop
        elif label == 'Weed':
            color = (0, 0, 255)  # Red for Weed
        else:
            color = (255, 0, 0)  # Blue for Unknown
        
        cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display the frame
        cv2.imshow('Crop and Weed Detection', frame)
        
        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the live detection function
detect_crop_weed_live()
