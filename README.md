# Gender-Detection-from-Hand-Palm-Images

This project aims to demonstrate various stages of a machine learning pipeline, including image preprocessing, feature extraction, and prediction using PySpark MLlib.

## Dataset available at: https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset
## Preprocessing Steps using Python
- Mount Google Drive to access data.
- Perform preprocessing on images, including:
  - Converting to grayscale.
  - Applying Difference of Gaussians (DoG) filter.
  - Normalizing and enhancing images.
  - Applying adaptive histogram equalization.
- Save preprocessed images in a designated folder.

## Feature Extraction from Images
- Extract features from preprocessed images using the VGG19 model.
- Save the extracted features to a CSV file.

## Using PySpark MLlib for Prediction
- Install PySpark.
- Load the extracted features from the CSV file.
- Prepare the data for training using Spark DataFrame.
- Train a Multilayer Perceptron Classifier (MLP) model.
- Evaluate the trained model on training and test datasets.
- Calculate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC.

## Usage
1. Mount Google Drive.
2. Preprocess images using the provided Python script.
3. Extract features from preprocessed images.
4. Install PySpark and run the PySpark script for prediction.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Pandas
- PySpark

## Note
Ensure proper configuration and access to data paths before running the scripts.
