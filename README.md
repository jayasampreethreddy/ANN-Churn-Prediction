# ANN Churn Prediction

This repository contains a project for predicting customer churn using an Artificial Neural Network (ANN) implemented with Keras and TensorFlow. The model is trained on a dataset that includes various customer features from a bank, and it predicts whether a customer will churn (leave the bank) or not.

## Project Overview

- **Objective**: To build an ANN model that predicts customer churn based on various features.
- **Dataset**: The dataset contains customer data, including features like geography, gender, credit score, balance, and whether or not the customer has exited the bank.
- **Tools and Libraries**:
  - Python 3.12.4
  - Pandas
  - NumPy
  - Matplotlib
  - Scikit-learn
  - Keras
  - TensorFlow

## Project Structure

- **ann.py**: The main Python script that preprocesses the data, builds the ANN model, trains it, and evaluates its performance.
- **Churn_Modelling.csv**: The dataset used for training and testing the model (ensure this file is placed in the same directory as `ann.py`).

## Data Preprocessing

- The dataset is imported using Pandas.
- Categorical variables like `Geography` and `Gender` are converted into dummy variables.
- The dataset is split into training and testing sets.
- Feature scaling is applied to ensure that all input features are on the same scale.

## ANN Model

- The model is built using the Keras `Sequential` API.
- **Input Layer**: 11 input features.
- **Hidden Layers**:
  - Two hidden layers with 6 nodes each and ReLU activation function.
- **Output Layer**:
  - A single node with a sigmoid activation function to predict the probability of churn.
  
- **Compilation**:
  - The model is compiled using the Adamax optimizer and binary crossentropy loss function.
  
- **Training**:
  - The model is trained on the training set with a validation split of 33% and a batch size of 10 for 100 epochs.
  
## Model Evaluation

- The accuracy and loss during training and validation are plotted.
- The model's performance is evaluated using a confusion matrix and accuracy score on the test set.

