"""
Plant Disease Prediction System - CNN Image Classifier

A Streamlit app for plant leaf disease classification using a trained
TensorFlow/Keras CNN model.

Model handling:
- The trained .h5 model is intentionally not committed to GitHub.
- The app first checks app/trained_model/plant_disease_prediction_model.h5.
- If the file is missing, it tries to download it from the GitHub Release URL below.

Run locally:
    streamlit run app/main.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "trained_model" / "plant_disease_prediction_model.h5"
CLASS_INDEX_PATH = APP_DIR / "class_indices.json"
TARGET_SIZE: Tuple[int, int] = (224, 224)

# This URL will work after you create the GitHub Release and upload the model asset.
DEFAULT_MODEL_URL = (
    "https://github.com/bhavyajain0810/plant-disease-cnn-classifier/"
    "releases/download/v1.0-model/plant_disease_prediction_model.h5"
)
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL", DEFAULT_MODEL_URL)


st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="centered",
)


def download_model(model_url: str, destination: Path) -> bool:
    """Download the model from a release URL if it is not already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")

    try:
        with requests.get(model_url, stream=True, timeout=30) as response:
            if response.status_code != 200:
                return False

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            progress = st.progress(0, text="Downloading trained model...")

            with temp_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress.progress(
                                min(downloaded / total_size, 1.0),
                                text="Downloading trained model...",
                            )

            temp_path.replace(destination)
            progress.empty()
            return True

    except requests.RequestException:
        if temp_path.exists():
            temp_path.unlink()
        return False


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load the trained Keras model once per Streamlit session."""
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_class_indices(class_index_path: str) -> Dict[int, str]:
    """Load class-index mapping and normalize JSON keys to integers."""
    with open(class_index_path, "r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    return {int(index): class_name for index, class_name in raw_mapping.items()}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert uploaded image into a normalized 224 x 224 RGB tensor."""
    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
    class_indices: Dict[int, str],
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Predict class label, confidence score, and sorted class probabilities."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_indices.get(predicted_index, f"Class {predicted_index}")
    confidence = float(predictions[predicted_index])

    probabilities = [
        (class_indices.get(index, f"Class {index}"), float(score))
        for index, score in enumerate(predictions)
    ]
    probabilities.sort(key=lambda item: item[1], reverse=True)

    return predicted_class, confidence, probabilities


def show_model_setup_message() -> None:
    """Display model setup instructions when download/local load is unavailable."""
    st.warning("Model file is missing and could not be downloaded automatically.")

    st.markdown(
        f"""
        Add the trained model manually at:

        ```text
        {MODEL_PATH}
        ```

        Or upload it as a GitHub Release asset named:

        ```text
        plant_disease_prediction_model.h5
        ```

        Expected release URL:

        ```text
        {MODEL_DOWNLOAD_URL}
        ```

        The model is intentionally excluded from Git commits because trained model
        artifacts are large binary files.
        """
    )


def main() -> None:
    st.title("🌿 Plant Disease Prediction System")
    st.caption("CNN image classifier built with TensorFlow/Keras and Streamlit.")

    st.markdown(
        """
        Upload a plant leaf image to classify it using a trained CNN model.
        The app resizes images to **224 × 224**, runs TensorFlow/Keras inference,
        and displays the predicted class with confidence scores.
        """
    )

    if not CLASS_INDEX_PATH.exists():
        st.error(f"Class mapping file not found: {CLASS_INDEX_PATH}")
        st.stop()

    if not MODEL_PATH.exists():
        with st.spinner("Model file not found locally. Trying to download from release..."):
            downloaded = download_model(MODEL_DOWNLOAD_URL, MODEL_PATH)

        if not downloaded:
            show_model_setup_message()
            st.stop()

    try:
        model = load_trained_model(str(MODEL_PATH))
        class_indices = load_class_indices(str(CLASS_INDEX_PATH))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error("Unable to load the model or class mapping.")
        st.exception(error)
        st.stop()

    uploaded_image = st.file_uploader(
        "Upload a leaf image",
        type=("jpg", "jpeg", "png"),
        help="Supported formats: JPG, JPEG and PNG",
    )

    if uploaded_image is None:
        st.info("Upload a plant leaf image to get a prediction.")
        return

    try:
        image = Image.open(uploaded_image)
    except UnidentifiedImageError:
        st.error("The uploaded file could not be read as an image.")
        return

    image_col, prediction_col = st.columns([1, 1])

    with image_col:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with prediction_col:
        st.subheader("Prediction")

        if st.button("Classify Image", type="primary"):
            predicted_class, confidence, probabilities = predict_image(
                model=model,
                image=image,
                class_indices=class_indices,
            )

            st.success(f"Predicted Class: {predicted_class}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            st.write("Class probabilities")
            for class_name, score in probabilities:
                st.progress(score, text=f"{class_name}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
