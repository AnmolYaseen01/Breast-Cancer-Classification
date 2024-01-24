# Breast Cancer Prediction

This repository contains a machine learning model for predicting breast cancer types (Malignant or Benign) based on various features. The dataset used for training and testing the model is the Breast Cancer Wisconsin (Diagnostic) dataset from the scikit-learn library.

## Overview

Breast Cancer Prediction is a machine learning project owned by Anmol Yaseen. The project is designed to predict the type of breast cancer (Malignant or Benign) based on various features. It utilizes the Breast Cancer Wisconsin (Diagnostic) dataset from the scikit-learn library. The primary goal is to build an accurate predictive model and create a user-friendly interface for users to input data and receive predictions.

## Dependencies

Make sure you have the following dependencies installed before running the code:

- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

You can install these dependencies using the following command:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Project Structure

The project consists of the following key components:

1. **Data Collection & Processing:**
   - The breast cancer dataset is loaded from scikit-learn and processed using pandas.
   - Features include mean radius, mean texture, mean perimeter, and more.
   - The dataset is split into features (X) and the target variable (Y).
   - Data is further split into training and testing sets for model development.

2. **Exploratory Data Analysis (EDA):**
   - EDA is performed to understand the characteristics of the dataset.
   - Visualizations such as pair plots, count plots, and heatmaps are used to explore feature relationships.
   - The distribution of the target variable is analyzed.

3. **Feature Scaling:**
   - Standard scaling is applied to normalize the features, ensuring consistency in model training.

4. **Model Training:**
   - A logistic regression model is implemented for breast cancer prediction.
   - The model is trained using the training data to learn patterns and relationships.

5. **Model Evaluation:**
   - The accuracy of the model is evaluated on both training and testing datasets.
   - Mean squared error and the coefficient of determination (R-squared) are calculated for testing data.

6. **Receiver Operating Characteristic (ROC) Curve:**
   - A ROC curve is plotted to visualize the performance of the model in distinguishing between Malignant and Benign cases.

7. **Building a Predictive System:**
   - A user-friendly interface allows users to input breast cancer feature values.
   - The trained model predicts the cancer type based on the provided input.

## Usage

To use the Breast Cancer Prediction system:

1. Ensure you have the required dependencies installed using `pip install numpy pandas scikit-learn seaborn matplotlib`.
2. Run the provided Python script.
3. Follow the on-screen prompts to input values for breast cancer features.
4. Receive the predicted cancer type (Malignant or Benign) based on the input.

## Further Contributions

This project is open to further contributions and improvements. Feel free to explore the code, suggest enhancements, or open issues for any questions or concerns. The aim is to create a robust and effective breast cancer prediction model while maintaining user accessibility and understanding.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/stable/) for providing the Breast Cancer Wisconsin (Diagnostic) dataset.
- The open-source community for creating and maintaining essential libraries used in this project.
