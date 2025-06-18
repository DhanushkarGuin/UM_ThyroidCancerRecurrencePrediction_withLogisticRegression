import pandas as pd
import pickle

pipeline = pickle.load(open('pipeline.pkl', 'rb'))

columns = ['Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

test_input = pd.DataFrame([[20, 'M', 'No', 'Yes', 'No', 'Euthyroid',
                            'Single nodular goiter-left', 'No',
                            'Micropapillary', 'Uni-Focal', 'Low',
                            'T1a', 'N0', 'M0', 'I', 'Excellent']], columns = columns)

prediction = pipeline.predict(test_input)
print('Recurrence:', 'No' if prediction == 0 else 'Yes')