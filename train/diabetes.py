import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

directory = os.getcwd()

# DATA FOR PRED
data=pd.read_csv(directory + "/dataset/diabetes.csv")

# Print Head of Dataset

print('\nDataFrame')
print("--------------------------------------------------------")
print(data.head())

# Check null values

print('\nNull Values')
print("--------------------------------------------------------")

print(data.isna().sum())

# Model
logreg=LogisticRegression(max_iter=500)

X=data.iloc[:,:8]

# Target Result
y=data[["Outcome"]]

X=np.array(X)

y=np.array(y)

# Split data between test & train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

# Train Model
logreg.fit(X_train, np.ravel(y_train,order='C'))

# Predict 
y_pred = logreg.predict(X_test)

# Accuracy
print('\nAccuracy : {}'.format(accuracy_score(y_test, y_pred)))

# Generate Report
clf_report = classification_report(y_test, y_pred)
print('\nClassification report')
print("--------------------------------------------------------")
print(clf_report)

joblib.dump(logreg, directory + "/models/diabetes")

