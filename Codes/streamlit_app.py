import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Define the custom functions required to load the model
def euclidean_distance(vects):
    """Calculates the euclidean distance between two vectors."""
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def contrastive_loss(y_true, y_pred):
    """Contrastive loss function needed for model compilation."""
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_siamese_model(model_path):
    """Loads the saved Keras model with its custom objects."""  
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "contrastive_loss": contrastive_loss,
                "euclidean_distance": euclidean_distance,
            },
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image_file):
    """
    Takes an uploaded image file, converts it to grayscale, resizes to 28x28,
    and formats it for the model.
    """
    # Open the image using Pillow
    img = Image.open(image_file)

    # Convert to grayscale
    img = img.convert("L")

    # Resize to 28x28, the input size for our MNIST model
    img = img.resize((28, 28))

    # Convert image to numpy array and normalize
    img_array = np.array(img) / 255.0

    # Reshape for the model: (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=[0, -1])

    return img_array


# Streamlit App UI

st.set_page_config(layout="wide")

st.title("🔢 Siamese Network for MNIST Digit Similarity")
st.write("Upload two images of handwritten digits (0-9) to see if they are the same.")

# Load the model
MODEL_PATH = "siamese_mnist_model.keras"
model = load_siamese_model(MODEL_PATH)

if model:
    # Create two columns for file uploaders
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image 1")
        uploaded_file1 = st.file_uploader(
            "Choose the first image...", type=["jpg", "jpeg", "png"]
        )

    with col2:
        st.subheader("Upload Image 2")
        uploaded_file2 = st.file_uploader(
            "Choose the second image...", type=["jpg", "jpeg", "png"]
        )

    # Prediction and Results

    if uploaded_file1 and uploaded_file2:
        # Preprocess both images
        image1 = preprocess_image(uploaded_file1)
        image2 = preprocess_image(uploaded_file2)

        # Display the uploaded images
        st.write("")
        st.subheader("Your Uploaded Images")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(uploaded_file1, caption="Image 1", use_column_width=True)
        with img_col2:
            st.image(uploaded_file2, caption="Image 2", use_column_width=True)

        # Make a prediction
        prediction = model.predict([image1, image2])
        distance = prediction[0][0]

        # Define a threshold
        SIMILARITY_THRESHOLD = 0.5

        # Display the results
        st.write("---")
        st.header("Results")
        st.metric(label="Predicted Distance", value=f"{distance:.4f}")

        if distance < SIMILARITY_THRESHOLD:
            st.success("✅ The model predicts these are the **SAME** digit.")
        else:
            st.error("❌ The model predicts these are **DIFFERENT** digits.")

        st.info(
            f"Note: A distance closer to 0 means the images are more similar. The decision threshold is set to {SIMILARITY_THRESHOLD}."
        )

else:
    st.warning(
        "Model file not found or could not be loaded. Please ensure `siamese_mnist_model.keras` is in the same directory."
    )
