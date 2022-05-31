import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection import cross_validate

import os

directory = os.getcwd()

# Read dataset
data = pd.read_csv(directory + "/dataset/heart.csv")

# Print Head of Dataset

print('\nDataFrame')
print("--------------------------------------------------------")
print(data.head())

# Perform Logarithm to scale down
data["trestbps"]=np.log(data["trestbps"])
data["chol"]=np.log(data["chol"])

# Drop unwanted columns/features
data=data.drop(["fbs"],axis=1)
data=data.drop(["ca"],axis=1)

# Target Result
target=data["target"]

# Check null values

print('\nNull Values')
print("--------------------------------------------------------")

print(data.isna().sum())

# Shuffel data to prevent bias
np.random.shuffle(data.values)

# Drop target
data=data.drop(["target"],axis=1)

# Scale down data
sc= StandardScaler()
data=sc.fit_transform(data)

# Logistic Regression
lr=LogisticRegression(max_iter=500)

# Split data between test & train
X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.3,random_state=123)

# Fit the Model according to data set
lr.fit(X_train, y_train)

# Perform Prediction
y_pred = lr.predict(X_test)

# Print Accuracy
print('\nAccuracy : {}'.format(accuracy_score(y_test, y_pred)))

# Generate Report
clf_report = classification_report(y_test, y_pred)
print('\nClassification report')
print("--------------------------------------------------------")
print(clf_report)

joblib.dump(lr, directory + "/models/heart")

