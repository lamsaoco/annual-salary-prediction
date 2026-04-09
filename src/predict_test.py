import pickle
import xgboost as xgb
import numpy as np

with open('ml_xgboost.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

cv = {'department': 'it',
 'job_title': 'software_engineer',
 'location': 'bosnia_and_herzegovina',
 'performance_rating': 'rating2',
 'experience_years': 4,
 'status': 'active',
 'work_mode': 'on-site'}

features = dv.get_feature_names_out().tolist()
X = dv.transform(cv)
d = xgb.DMatrix(X, feature_names=features)
result = model.predict(d)
result = result[0]

print(f'The candidate’s predicted annual salary is {format(round(np.expm1(result), 0), ".0f")} VND')

