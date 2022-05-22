import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

directory = os.getcwd()

# DATA FOR PRED
data=pd.read_csv(directory + "/dataset/diabetes.csv")

print(data.head())

logreg=LogisticRegression(max_iter=500)

X=data.iloc[:,:8]

print(X.shape[1])

y=data[["Outcome"]]

X=np.array(X)

y=np.array(y)

logreg.fit(X,y.reshape(-1,))

joblib.dump(logreg, directory + "/models/diabetes")

