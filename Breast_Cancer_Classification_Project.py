

""" Importing the Dependencies"""

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection & Processing"""

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

print(breast_cancer_dataset)

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# print the first 5 rows of the dataframe
data_frame.head()

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# print last 5 rows of the dataframe
data_frame.tail()

# number of rows and columns in the dataset
data_frame.shape

# getting some information about the data
data_frame.info()

# checking for missing values
data_frame.isnull().sum()

# statistical measures about the data
data_frame.describe()

# checking the distribution of Target Varibale
data_frame['label'].value_counts()

"""1 --> Benign

0 --> Malignant
"""

data_frame.groupby('label').mean()

"""# Statistical Analysis of the Data"""

description=data_frame.describe()
print(f"description of the dataset: ")
print(description)

corr=data_frame.corr()
corr

std=data_frame.std()
print(f"std: {std}")

"""# Exploratory Data Analysis"""

import seaborn as sns
import matplotlib.pyplot as plt

# visualize the data

# Features with higher standard deviations
selected_features = ['area error', 'worst perimeter', 'worst area', 'mean area', 'mean perimeter']

# Adding 'label' as hue to differentiate classes
sns.pairplot(data_frame, vars=selected_features, hue='label', markers=["o", "s"], palette="husl")
plt.show()

import seaborn as sns
sns.countplot(x='label',data=data_frame)

sns.distplot(data_frame['label'])

sns.displot(data_frame['label'])

correlation = data_frame.corr()

# constructing a Heat Map
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

"""Separating the features and target"""

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

print(X)

print(Y)

"""Splitting the data into training data & Testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

"""To check if the number of rows in x_train matches the number of elements in y_train. You can use x_train.shape[0] to get the number of rows in x_train and compare it with len(y_train)."""

print(X_train.shape)
print(Y_train.shape)
print(X_train.shape[0] == Y_train.shape[0])
print(X_train.shape == Y_train.shape)

print(X_test.shape)         # print(len(X_test))
print(Y_test.shape)          #print(len(Y_test))
print(X_test.shape[0] == Y_test.shape[0])
print(X_test.shape == Y_test.shape)

print("Missing values in y_train:")
print(Y_train.isnull().sum())
print("\nMissing values in y_test:")
print(Y_test.isnull().sum())
print("\nUnique values in y_test:")
print(Y_train.unique())

# Check indices of x_train and y_train
print(X_train.index)
print(Y_train.index)

"""# Feature Scaling"""

from sklearn.preprocessing import StandardScaler

# Scale the training data
scaler = StandardScaler().fit(X_train)
x_train = scaler.transform(X_train)

"""# Model Training

Implementing Logistic Regression Model
"""

model = LogisticRegression()

# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)

"""# Model Evaluation

# Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

from sklearn.metrics import mean_squared_error,r2_score

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, X_test_prediction))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, X_test_prediction))

"""# Reciever Operating Curve"""

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(Y_test, X_test_prediction)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

"""# Building a Predictive System"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Replace the original feature names with the provided list
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def preprocess_input(user_input):
    """
    Preprocess the user input before making predictions.
    """

    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(user_input.reshape(1, -1))
    return scaled_input

def predict_cancer_type(model, user_input):
    """
    Make predictions using the trained machine learning model.
    """
    preprocessed_input = preprocess_input(user_input)
    prediction = model.predict(preprocessed_input)
    return prediction

def get_user_input():
    """
    Get input from the user for breast cancer features with validation.
    """
    user_input = np.zeros(len(feature_names) - 1)  # Assuming you have 30 features excluding the label

    # Get input from the user for each feature with validation
    for i, feature_name in enumerate(feature_names[:-1]):  # Exclude the label
        while True:
            try:
                user_input[i] = float(input(f"Enter value for {feature_name}: "))
                break  # Break the loop if the input is valid
            except ValueError:
                print("Invalid input. Please enter a valid numerical value.")

    return user_input

def main():
    # Get input from the user
    user_input = get_user_input()

    # Make a prediction
    prediction = predict_cancer_type(model, user_input)

    # Display the prediction label
    if prediction == 1:
        print("The predicted cancer type is: Malignant")
    else:
        print("The predicted cancer type is: Benign")

if __name__ == "__main__":
    main()