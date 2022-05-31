import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

directory = os.getcwd()

# Read dataset
patients=pd.read_csv(directory + "/dataset/liver.csv")

print('\nDataFrame')
print("--------------------------------------------------------")

# Print Head of Dataset
print(patients.head())

# Convert to numerical
patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)

# Check null values

print('\nNull Values')
print("--------------------------------------------------------")

print(patients.isna().sum())

# Fill null values
patients["Albumin_and_Globulin_Ratio"].fillna(patients['Albumin_and_Globulin_Ratio'].mean(), inplace = True)

# Drop features
X=patients[['Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]

# Target Outcome
y=patients['Dataset']

# Split data between test & train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

model = ensemble.RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('\nAccuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('\nClassification report')
print("--------------------------------------------------------")
print(clf_report)

joblib.dump(model,directory + "/models/liver")



