import pickle
from pathlib import Path
import xgboost as xgb
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

# Resolve model path relative to project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ml_xgboost.bin"

class CVRecords(BaseModel):
    department: Literal["it", "sales", "operations", "marketing", "finance", "hr", "r&d"]
    job_title: Literal[
        "software_engineer",
        "sales_executive",
        "operations_executive",
        "account_manager",
        "marketing_executive",
        "data_analyst",
        "accountant",
        "devops_engineer",
        "logistics_coordinator",
        "hr_executive",
        "seo_specialist",
        "business_development_manager",
        "financial_analyst",
        "it_manager",
        "research_scientist",
        "talent_acquisition_specialist",
        "supply_chain_manager",
        "content_strategist",
        "cto",
        "finance_manager",
        "product_developer",
        "hr_manager",
        "sales_director",
        "operations_director",
        "lab_technician",
        "brand_manager",
        "cfo",
        "hr_director",
        "innovation_manager"
    ]
    location: Literal[
        "korea",
        "congo",
        "bouvet_island_(bouvetoya)",
        "western_sahara",
        "iceland",
        "lebanon",
        "palestinian_territory",
        "montenegro",
        "saint_helena",
        "cook_islands"
    ]
    performance_rating: Literal["rating1",'rating2','rating3','rating4','rating5']
    experience_years: int = Field(..., ge=0)
    status: Literal[
        "active",
        "resigned",
        "retired",
        "terminated"
    ]
    work_mode: Literal["on-site","remote"]

class PredictResponse(BaseModel):
    annual_salary: int

app = FastAPI(title="annual-salary-prediction")

with open(MODEL_PATH, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

# cv = {'department': 'it',
#  'job_title': 'software_engineer',
#  'location': 'bosnia_and_herzegovina',
#  'performance_rating': 'rating2',
#  'experience_years': 4,
#  'status': 'active',
#  'work_mode': 'on-site'}

def predict_single(cv):
    features = dv.get_feature_names_out().tolist()
    X = dv.transform(cv)
    d = xgb.DMatrix(X, feature_names=features)
    result = model.predict(d)
    result = result[0]
    return round(np.expm1(result), 0)

@app.post("/predict")
def predict(cv: CVRecords) -> PredictResponse:
    prob = predict_single(cv.model_dump())
    
    return PredictResponse(
        annual_salary=prob
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

#print(f'The candidate’s predicted annual salary is {format(round(np.expm1(result), 0), ".0f")} VND')

