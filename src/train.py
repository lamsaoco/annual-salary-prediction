import pickle
from pathlib import Path

# Resolve paths relative to project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ml_xgboost.bin"

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

def load_data():
    # Set the path to the file you'd like to load
    file_path = "HR_Data_MNC_Data Science Lovers.csv"

    # Load the latest version
    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "rohitgrewal/hr-data-mnc",
    file_path,
    )
    df = df.sample(frac=0.3, random_state=42)
    df = df.reset_index(drop=True)
    df = df.drop(['Unnamed: 0', 'Employee_ID', 'Full_Name'], axis=1)
    df.columns = df.columns.str.lower().str.replace(' ','_')

    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ','_')

    df['location'] = df['location'].str.split(',_').str[-1]
    df['salary_vnd'] = round(df['salary_inr'] * 296.77, 0)

    del df['salary_inr']
    del df['hire_date']

    performance_rating_values = {
        1: 'rating1',
        2: 'rating2',
        3: 'rating3',
        4: 'rating4',
        5: 'rating5'
    }
    df.performance_rating = df.performance_rating.map(performance_rating_values)

    return df


def train_model(df):
    y_train = np.log1p(df['salary_vnd'])
    del df['salary_vnd']

    train_dicts = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dicts)
    X_train = dv.transform(train_dicts)
    features = dv.get_feature_names_out().tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    xgb_params = {
            'eta': 0.3, 
            'max_depth': 10,
            'min_child_weight': 1,      
            'objective': 'reg:squarederror',
            'nthread': 8,
            'eval_metric': 'rmse',     
            'seed': 42,
            'verbosity': 1,
        }
    model = xgb.train(xgb_params, dtrain, num_boost_round=81, verbose_eval=5)

    return dv, model

def save_model(dv, model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

df = load_data()
dv, model = train_model(df)
save_model(dv, model, MODEL_PATH)

print(f'Model saved to {MODEL_PATH}')

