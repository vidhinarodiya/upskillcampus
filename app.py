import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to RGB (since VGG16 expects 3 channels)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (128, 128))
    normalized = resized / 255.0
    return normalized

# Function to load and predict using the model
def predict(image, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    return prediction[0][0]

# Load the model
model_path = 'crop_weed_detection_model.h5'  # Replace with your model path
model = load_model(model_path)

# Streamlit app
def main():
    st.title('Crop and Weed Detection')
    st.write('Upload an image to detect if it is a Crop or a Weed.')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpeg', 'jpg', 'png'])

    if uploaded_file is not None:
        try:
            # Read image bytes
            image_bytes = uploaded_file.read()
            # Convert to OpenCV format
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Predict the result
            if st.button('Detect'):
                result = predict(image, model)
                if result > 0.8:
                    st.success('Prediction: Crop')
                else:
                    st.success('Prediction: Weed')

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
