# Plant Disease Prediction System

A Streamlit-based plant disease prediction application that classifies apple leaf images using a trained TensorFlow/Keras CNN model.

The current version is configured for a binary classification demo using the provided `class_indices.json` mapping:

* Apple Scab
* Healthy

## Features

* Upload plant leaf images through a Streamlit web interface
* Preprocess images to `224 x 224` RGB format
* Run inference using a trained CNN model
* Display predicted class, confidence score, and class probabilities
* Keep large trained model files and datasets outside normal Git history
* Includes a Jupyter notebook for the model-development workflow
* Docker-ready Streamlit app structure

## Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pillow
* Requests
* Jupyter Notebook
* Docker

## Project Structure

```text
plant-disease-cnn-classifier/
├── app/
│   ├── main.py
│   ├── class_indices.json
│   ├── requirements.txt
│   ├── Dockerfile
│   └── config.toml
├── model_training_notebook/
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The model-training workflow is based on the PlantVillage dataset available on Kaggle:

```text
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```

The full dataset is not included in this repository. Download it separately if you want to retrain or extend the model.

## Model File

The trained `.h5` model file is not committed to GitHub because it is a large binary artifact.

For local use, place the trained model at:

```text
app/trained_model/plant_disease_prediction_model.h5
```

The app is configured to look for this file locally. If it is not present, it can download the model from a GitHub Release asset after the release is created.

Release asset URL:

```text
https://github.com/bhavyajain0810/plant-disease-cnn-classifier/releases/download/v1.0-model/plant_disease_prediction_model.zip
```

## Local Setup

Clone the repository:

```bash
git clone https://github.com/bhavyajain0810/plant-disease-cnn-classifier.git
cd plant-disease-cnn-classifier
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the model directory:

```powershell
New-Item -ItemType Directory -Path app\trained_model -Force
```

Place the trained model here:

```text
app/trained_model/plant_disease_prediction_model.h5
```

Run the Streamlit app:

```bash
streamlit run app/main.py
```

Open the local app in your browser:

```text
http://localhost:8501
```

## Docker Setup

From the `app/` directory:

```bash
cd app
docker build -t plant-disease-predictor .
docker run -p 8501:8501 plant-disease-predictor
```

Then open:

```text
http://localhost:8501
```

## Model Workflow

The training notebook covers the basic image-classification pipeline:

1. Dataset loading and inspection
2. Image preprocessing
3. Train/validation split setup
4. CNN model creation
5. Model training and evaluation
6. Model export for Streamlit inference

## Local Model Check

Two trained model artifacts were compared on local Apple Scab and Healthy test samples. The final selected model achieved:

```text
Apple Scab: 11/11 correct
Healthy: 12/12 correct
Overall: 23/23 correct
Average confidence: 95.17%
```

This is a small local validation sample and should not be interpreted as full benchmark accuracy on the complete PlantVillage dataset.

## Current Scope

This repository currently supports binary apple leaf classification:

* Apple Scab
* Healthy

To extend the project into a larger multi-class PlantVillage classifier, retrain the model on additional disease classes and update:

```text
app/class_indices.json
```

## Files Excluded from Git

The following files and folders are intentionally excluded:

```text
venv/
app/venv/
app/trained_model/
*.h5
*.keras
*.pkl
*.pt
*.pth
*.onnx
dataset/
datasets/
data/
test_images/
credentials.toml
secrets.toml
```
