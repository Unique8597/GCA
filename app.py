import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from preprocess import preprocess_image  # ensure this returns a (1, 224, 224, 3) tensor

token = st.secrets["HF_TOKEN"]

@st.cache_resource
def load_model_from_hub():
    # Download model file from Hugging Face Hub
    model_path = hf_hub_download(
    repo_id="Unique8597/my-keras-model",
    filename="resnet_garbage_classifier.keras",
    token=token
)
    model = tf.keras.models.load_model(model_path)
    return model
# ==============================
# 🔹 Load Trained Model
# ==============================
model = load_model_from_hub()
st.success("✅ Model loaded successfully!")

# ==============================
# 🔹 Class Labels
# ==============================
class_names = [
    'brown-glass',
    'cardboard',
    'green-glass',
    'metal',
    'paper',
    'plastic',
    'trash',
    'white-glass'
]

# ==============================
# 🔹 App UI
# ==============================
st.title("🗑️ RIN - Recycle Image Notifier")
st.write("Upload or capture an image to classify the waste type.")

mode = st.radio("Choose input method:", ["📁 Upload Image", "📷 Use Camera"])

# ==============================
# 🔹 Image Upload Mode
# ==============================
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess & Predict
        img_tensor = preprocess_image(image)
        prediction = model.predict(img_tensor)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display Results
        st.success(f"🧩 Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

# ==============================
# 🔹 Camera Mode
# ==============================
elif mode == "📷 Use Camera":
    picture = st.camera_input("Take a photo")
    if picture:
        image = Image.open(picture).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        # Preprocess & Predict
        img_tensor = preprocess_image(image)
        prediction = model.predict(img_tensor)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display Results
        st.success(f"🧩 Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
