# Task 2 — Deep Learning Model for Image Classification (TensorFlow/Keras)

## Objective
Implement a functional deep learning model for an image classification task using **TensorFlow/Keras**, and present the **visualization of results**. This deliverable demonstrates competency in building, training, and evaluating a CNN within a reproducible ML workflow.

## Overview
- **Dataset**: `sklearn.datasets.load_digits` (8×8 grayscale images, 10 classes).  
  This choice ensures offline reproducibility without external downloads.
- **Model**: Compact **Convolutional Neural Network (CNN)** with two convolutional layers, max-pooling, a dense hidden layer, dropout regularization, and a softmax output.
- **Metrics**: Accuracy and cross-entropy loss (training, validation, and test).
- **Visualizations**:
  - Training vs Validation **Accuracy** (`training_accuracy.png`)
  - Training vs Validation **Loss** (`training_loss.png`)
  - **Confusion Matrix** on the test set (`confusion_matrix.png`)
  - **Sample Predictions** grid with predicted vs true labels (`sample_predictions.png`)
- **Artifacts**:
  - Trained model: `digits_cnn.keras`
  - Report: `classification_report.txt`

## Repository Structure (Suggested)
task2/
├─ task2_model.py
├─ README.md
└─ artifacts/ # created automatically after training
├─ training_accuracy.png
├─ training_loss.png
├─ confusion_matrix.png
├─ sample_predictions.png
├─ classification_report.txt
└─ digits_cnn.keras


## Setup
**Python Version**: 3.9–3.12  
**Dependencies**:
- tensorflow (or tensorflow-cpu)
- scikit-learn
- matplotlib
- numpy

Install:
```bash
pip install tensorflow-cpu scikit-learn matplotlib numpy
