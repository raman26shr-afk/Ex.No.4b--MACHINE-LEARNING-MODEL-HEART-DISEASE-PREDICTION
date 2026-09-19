# Ex.No.4b--MACHINE-LEARNING-MODEL-HEART-DISEASE-PREDICTION

# AIM

To develop a Heart Disease Prediction model using machine learning classification algorithms and compare the performance of different models using suitable evaluation metrics.

# OBJECTIVES
To understand machine learning classification.
To analyze a heart disease dataset.
To identify input features and the target variable.
To preprocess the dataset.
To divide the dataset into training and testing data.
To train different classification models.
To predict whether a patient has heart disease.
To evaluate and compare the performance of different models.

# INTRODUCTION

Machine Learning enables computers to learn patterns from data and make predictions without being explicitly programmed for every situation.

Classification is a supervised learning technique used to predict a category or class. In this experiment, classification algorithms are used to predict whether a patient is likely to have heart disease.

The target variable generally contains two classes:

0 → No Heart Disease
1 → Heart Disease

Note: This experiment is for demonstrating machine-learning classification. A model prediction should not be treated as a medical diagnosis.

# THEORY

Heart disease prediction is a binary classification problem. Patient information such as age, blood pressure, cholesterol, chest pain type, and maximum heart rate can be used as input features.

The machine-learning model learns relationships between these features and the target class from the training data. After training, the model predicts the class of previously unseen test data.

# DATA PREPROCESSING
Steps
Load the dataset.
Display the first few records.
Check the dataset shape.
Check data types.
Check missing values.
Handle missing values if present.
Separate input features and target.
Encode categorical variables if required.
Split the dataset into training and testing data.
Apply feature scaling where required.


# PROGRAM
Step 1: Import Libraries
im

port pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
Step 2: Load the Dataset
df = pd.read_csv("heart_disease.csv")

print("First five records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())
Step 3: Separate Input and Output
X = df.drop("target", axis=1)
y = df["target"]

print("Input features:")
print(X.head())

print("\nTarget:")
print(y.head())

Here:

X → Patient characteristics/input features
y → Heart disease prediction/target
Step 4: Train-Test Split

The dataset is divided into 80% training data and 20% testing data.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
Step 5: Feature Scaling

Feature scaling brings numerical features to a comparable scale.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

fit_transform() is applied only to the training data, while transform() is used for the test data.

# MACHINE LEARNING MODELS

1. Logistic Regression
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
2. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
3. Decision Tree
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
4. Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
5. Support Vector Machine
svm = SVC(kernel="rbf")

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
6. Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

# MODEL EVALUATION

The models are evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Evaluation Metrics

Accuracy measures the proportion of correctly classified samples.

Precision measures how many of the samples predicted as positive are actually positive.

Recall measures how many of the actual positive cases are correctly identified.

F1-score is the harmonic mean of precision and recall.

# Evaluation Code
models = {
    "Logistic Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "Gradient Boosting": y_pred_gb
}

results = []

for name, prediction in models.items():

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, zero_division=0)
    recall = recall_score(y_test, prediction, zero_division=0)
    f1 = f1_score(y_test, prediction, zero_division=0)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(name)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print()

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("MODEL COMPARISON")
print(results_df)

# EXPECTED OUTPUT
Dataset Information
First five records:
[First five rows of heart_disease.csv]

Dataset shape:
(number_of_rows, number_of_columns)

Missing values:
[Number of missing values in each column]

# APPLICATIONS
Medical decision-support systems
Risk assessment
Patient data analysis
Healthcare analytics
Clinical research
Early identification of potentially high-risk cases


# RESULT

Thus, different machine-learning classification algorithms were successfully implemented for heart disease prediction. The models were trained using patient-related features and evaluated using Accuracy, Precision, Recall, and F1-score. A confusion matrix was also generated to analyze the classification results.
