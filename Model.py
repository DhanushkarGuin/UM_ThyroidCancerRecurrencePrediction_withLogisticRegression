import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
#print(dataset)
dataset['Gender'] = dataset['Gender'].map({'F': 0, 'M': 1})
dataset['Focality'] = dataset['Focality'].map({'Uni-Focal': 0, 'Multi-Focal': 1})
dataset['M'] = dataset['M'].map({'M0': 0, 'M1':1})

binary_columns = ['Smoking', 'Hx Smoking', 'Hx Radiothreapy','Recurred']
for col in binary_columns:
    dataset[col] = dataset[col].map({'No': 0, 'Yes': 1})
#print(dataset)

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [5,6,7,8,10,11,12,14,15])], 
                    remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy %:", (accuracy_score(Y_test, Y_pred))*100)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))







