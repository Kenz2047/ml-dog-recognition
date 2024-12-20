# 🐾 Dog Breed Classification and Localization Project

Welcome to the **Dog Breed Classification and Localization Project**, developed as part of the **CS-233: Introduction to Machine Learning** course. 
This project implements machine learning methods to classify dog breeds and locate the center of dogs in images using the Stanford Dogs dataset.

## 📚 Project Overview

### Objectives
1. **Dog Breed Classification**: Identify the breed of a dog from an image.
2. **Dog Localization**: Predict the center point of a dog within an image.

The project is divided into two milestones:
- **Milestone 1**: Implementation of linear regression, logistic regression, and k-Nearest Neighbors (k-NN) for classification and regression tasks.
- **Milestone 2**: Development of deep learning models (Transformer and CNN) using PyTorch.

---

## 🚀 Features

### Milestone 1
- **Linear Regression** for locating the dog's center (regression).
- **Logistic Regression** for classifying dog breeds.
- **k-Nearest Neighbors (k-NN)** for both classification and regression tasks.

### Milestone 2
- **Transformer Model**: Uses attention mechanisms for image classification.
- **Convolutional Neural Network (CNN)**: Processes images for classification and localization.

---
## 📂 Dataset

### Stanford Dogs Dataset
- **20 Classes**: Dog breeds.
- **Feature Extraction**: Images preprocessed using ResNet-50 to reduce computational complexity.
- **Split**:
  - Training: 2,938 images.
  - Testing: 327 images.

### Data Structure
- Train Features: `(2938, 2048)`
- Test Features: `(327, 2048)`
- Train Labels: `(2938,)`
- Test Labels: `(327,)`
- Train Centers: `(2938, 2)`
- Test Centers: `(327, 2)`

---

## 📊 Evaluation Metrics

- **Classification**:
  - **Accuracy**: Percentage of correct predictions.
  - **F1-Score**: Macro-average F1-score across all classes.
- **Regression**:
  - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and true values.

---

## ⚙️ How to Run

### Milestone 1
# Linear Regression for Dog Localization
python main.py --method linear_regression --lamda 1 --task center_locating

# Logistic Regression for Dog Breed Classification
python main.py --method logistic_regression --lr 1e-5 --max_iters 100 --task breed_identifying

# k-Nearest Neighbors (k-NN) for Both Tasks
python main.py --method knn --K 1

### Milestone 2
# Transformer Model
python main2.py --method nn --nn_type transformer --lr 1e-5 --max_iters 100

# CNN Model
python main2.py --method nn --nn_type cnn --lr 1e-5 --max_iters 100

---

## 🧪 Testing

### Test Scripts
- **Milestone 1**: Verify functionality with `test_ms1.py`:
  python test_ms1.py
- **Milestone 2**: Verify functionality with `test_ms2.py`:
  python test_ms2.py
  
---

## 📝 Reports

Each milestone includes a concise 2-page report covering:
- **Introduction**: Project goals and objectives.
- **Methods**: Implementation details and hyperparameter tuning.
- **Experiments/Results**: Performance evaluation with visualizations.
- **Discussion/Conclusion**: Analysis of results and insights.

---

## 🛡️ Important Notes

- Preprocessing steps, such as normalization or data augmentation, are handled in `main.py`.
- Methods generalize to datasets of varying shapes and sizes.
