import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
# print(dataset)
dataset['Gender'] = dataset['Gender'].map({'F': 0, 'M': 1})
dataset['Focality'] = dataset['Focality'].map({'Uni-Focal': 0, 'Multi-Focal': 1})
dataset['M'] = dataset['M'].map({'M0': 0, 'M1':1})

binary_columns = ['Smoking', 'Hx Smoking', 'Hx Radiothreapy','Recurred']
for col in binary_columns:
    dataset[col] = dataset[col].map({'No': 0, 'Yes': 1})
# print(dataset)

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [5,6,7,8,10,11,12,14,15])], 
                    remainder='passthrough')
X = np.array(ct.fit_transform(X))

encoded_col_names = ct.named_transformers_['encoder'].get_feature_names_out()
non_encoded_col_names = [col for i, col in enumerate(dataset.columns[:-1]) if i not in [5,6,7,8,10,11,12,14,15]]
all_col_names = list(encoded_col_names) + non_encoded_col_names

X_df = pd.DataFrame(X, columns=all_col_names)
# print(X_df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
# No need of regularization here, the metrics of evaluation remain approximately same
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
# print('Predictions:', Y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print("Accuracy %:", (accuracy_score(Y_test, Y_pred))*100)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Precision:", (precision_score(Y_test, Y_pred)))
print("Recall:", recall_score(Y_test, Y_pred))

print("Please enter details to check your prediction!")
age = int(input("Enter Age in Years:"))
gender = input("Enter gender(F for Female and M for Male):")
smoking = input("Do you smoke now?:")
hx_smoking = input("Do you used to smoke?:")
hx_radiotherapy = input("Ever had Radiotherapy treatment?:")
thyroid_function = input("Status of Thyroid:")
physical_examination = input("Findings from a physical examination:")
adenopathy = input("Presence of adenopathy:")
pathology = input("Type of Thyroid detected in pathological examination:")
focality = input("Is cancer uni-focal or multi-focal?:")
risk = input("Category of the risk?:")
t = input("Type of Tumor:")
n = input("Type of node:")
m = input("Type of metastasis:")
stage = input("Overall stage of cancer:")
response = input("Response to the treatment:")

gender = 1 if gender == 'M' else 0
focality = 1 if focality == 'Multi-Focal' else 0
m = 1 if m == 'M1' else 0
smoking = 1 if smoking == 'Yes' else 0
hx_smoking = 1 if hx_smoking == 'Yes' else 0
hx_radiotherapy = 1 if hx_radiotherapy == 'Yes' else 0

thyroid_function_encoded = [0,0,0,0,0]
if thyroid_function == 'Clinical Hyperthyroidism':
    thyroid_function_encoded[0] = 1
elif thyroid_function == 'Clinical Hypothyroidism':
    thyroid_function_encoded[1] = 1
elif thyroid_function == 'Euthyroid':
    thyroid_function_encoded[2] = 1
elif thyroid_function == 'Subclinical Hyperthyroidism':
    thyroid_function_encoded[3] = 1
else:
    thyroid_function_encoded[4] = 1

physical_examination_encoded = [0,0,0,0,0]
if physical_examination == 'Diffuse goiter':
    physical_examination_encoded[0] = 1
elif physical_examination == 'Multinodular goiter':
    physical_examination_encoded[1] = 1
elif physical_examination == 'Normal':
    physical_examination_encoded[2] = 1
elif physical_examination == 'Single nodular goiter-left':
    physical_examination_encoded[3] = 1
else:
    physical_examination_encoded[4] = 1

adenopathy_encoded = [0,0,0,0,0,0]
if adenopathy == 'Bilateral':
    adenopathy_encoded[0] = 1
elif adenopathy == 'Extensive':
    adenopathy_encoded[1] = 1
elif adenopathy == 'Left':
    adenopathy_encoded[2] = 1
elif adenopathy == 'No':
    adenopathy_encoded[3] = 1
elif adenopathy == 'Posterior':
    adenopathy_encoded[4] = 1
else:
    adenopathy_encoded[5] = 1

pathology_encoded = [0,0,0,0]
if pathology == 'Follicular':
    pathology_encoded[0] = 1
elif pathology == 'Hurthel cell':
    pathology_encoded[1] = 1
elif pathology == 'Micropapillary':
    pathology_encoded[2] = 1
else:
    pathology_encoded[3] = 1

risk_encoded = [0,0,0]
if risk == 'High':
    risk_encoded[0] = 1
elif risk == 'Intermediate':
    risk_encoded[1] = 1
else:
    risk_encoded[2] = 1

t_encoded = [0,0,0,0,0,0,0]
if t == 'T1a':
    t_encoded[0] = 1
elif t == 'T1b':
    t_encoded[1] = 1
elif t == 'T2':
    t_encoded[2] = 1
elif t == 'T3a':
    t_encoded[3] = 1
elif t == 'T3b':
    t_encoded[4] = 1
elif t == 'T4a':
    t_encoded[5] = 1
else:
    t_encoded[6] = 1

n_encoded = [0,0,0]
if n == 'N0':
    n_encoded[0] = 1
elif n == 'N1a':
    n_encoded[1] = 1
else:
    n_encoded[2] = 1

stage_encoded = [0,0,0,0,0]
if stage == 'I':
    stage_encoded[0] = 1
elif stage == 'II':
    stage_encoded[1] = 1
elif stage == 'III':
    stage_encoded[2] = 1
elif stage == 'IIII':
    stage_encoded[3] = 1
else:
    stage_encoded[4] = 1

response_encoded = [0,0,0,0]
if response == 'Biochemical Incomplete':
    response_encoded[0] = 1
elif response == 'Excellent':
    response_encoded[1] = 1
elif response == 'Indeterminate':
    response_encoded[2] = 1
else:
    response_encoded[3] = 1

user_input = thyroid_function_encoded + physical_examination_encoded + adenopathy_encoded + pathology_encoded + risk_encoded + t_encoded + n_encoded + stage_encoded + response_encoded + [age, gender, smoking, hx_smoking, hx_radiotherapy, focality, m]

user_input = np.array(user_input).reshape(1, -1)

prediction = model.predict(user_input)
print("Recurrence Risk:", "Yes" if prediction[0] == 1 else "No")