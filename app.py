import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato_classifier_model.h5")

model = load_model()

# Define class names
class_names = ["Reject", "Ripe", "Unripe"]

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize image to match model input size
    image = img_to_array(image)      # Convert to array
    image = image / 255.0            # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Tomato Classifier")
st.write("Classify tomatoes as Reject, Ripe, or Unripe.")

# Option to upload an image or take a picture
option = st.radio("Choose an input method:", ["Upload an image", "Take a picture"])

uploaded_file = None

if option == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

elif option == "Take a picture":
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    # Display the uploaded or captured image
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", width = 300)
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display results
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
