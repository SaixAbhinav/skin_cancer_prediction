# Skin Cancer Prediction

This project leverages deep learning techniques to predict skin cancer from dermoscopy images. By using convolutional neural networks (CNNs) , the model distinguishes between lesions, providing an automated aid for early detection and diagnosis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Training & Evaluation](#model-training--evaluation)
---

## Overview

Early diagnosis of skin cancer is crucial for effective treatment. This project uses a deep learning approach to automatically analyze dermoscopy images and classify them. The goal is to assist healthcare professionals by providing a second opinion and expediting the diagnostic process.

---

## Features

- **Image Preprocessing:**  
  Resizing, normalization, and augmentation techniques to improve model generalization.
  
- **CNN Architecture:**  
  Implementation of a convolutional neural network built from scratch.
  
- **Model Evaluation:**  
  Utilizes metrics such as accuracy, precision, recall, F1-score, and confusion matrix to evaluate model performance.
  
- **Prediction Interface:**  
  A simple script for making predictions on new images.

---

## Dataset

This project uses a dataset of dermoscopy images ([HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)) . The dataset is preprocessed to extract relevant features for model training and validation.

- **Training Set:** Images with corresponding labels (malignant/benign).
- **Validation/Test Set:** A subset of images reserved to evaluate the performance of the model.

---

## Technologies Used

- **Programming Language:** Python 3.x
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Processing:** NumPy, Pandas
- **Image Processing:** OpenCV, PIL
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook

---

## Model Training & Evaluation
Training Process:
The model is trained using a combination of data augmentation and regularization techniques to reduce overfitting. Early stopping and model checkpointing are implemented to save the best-performing model.

Evaluation Metrics:

- **Accuracy:** Overall correctness of the model.
- **Precision & Recall:** For assessing the quality of malignant lesion predictions.
- **F1-Score:** The harmonic mean of precision and recall.
- **Confusion Matrix:** Visual representation of model predictions vs. actual labels.

---
