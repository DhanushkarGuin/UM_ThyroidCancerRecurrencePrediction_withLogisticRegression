import pandas as pd

dataset = pd.read_csv('dataset.csv')

def printData(number):
    print(dataset.head(number))

# printData(5)

dataset['Recurred'] = dataset['Recurred'].map({'Yes': 1, 'No':0})

X = dataset.drop(columns = 'Recurred')
y = dataset['Recurred']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

categorical_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function',
                        'Physical Examination', 'Adenopathy', 'Pathology',
                        'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

preprocess = ColumnTransformer([
    ('encode', OneHotEncoder(sparse_output=False,handle_unknown='ignore'), categorical_columns)
],remainder='passthrough')

pipeline = Pipeline([
    ('encode', preprocess),
    ('logreg', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import precision_score,recall_score,confusion_matrix
print('Precision:', precision_score(y_test,y_pred))
print('Recall:', recall_score(y_test,y_pred))
print('Confusion Matrix:', confusion_matrix(y_test,y_pred))

import pickle
pickle.dump(pipeline, open('pipeline.pkl', 'wb'))

# print(dataset.columns.tolist())