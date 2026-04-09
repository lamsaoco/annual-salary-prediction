***ML-Zoomcamp Midterm Project***

# Annual Salary Prediction

[![Python](https://img.shields.io/badge/python-3.12.10-blue)](https://www.python.org/)

This project provides a machine learning solution to predict a candidate's annual salary based on publicly available profile information, assisting HR teams in making data-driven and informed compensation decisions.

## Problem Statement

In recruitment, HR professionals often encounter candidates with diverse profiles. After passing multiple interview rounds, candidates reach the salary negotiation stage. Many candidates choose not to disclose their previous salary, creating challenges:

- Offering too low a salary might cause the candidate to decline.
- Offering too high may exceed the company's budget.

This project predicts a candidate's annual salary using key features such as department, job title, years of experience, location, performance rating, and more.

## Dataset

- **Source:** [Kaggle - HR Data MNC](https://www.kaggle.com/datasets/rohitgrewal/hr-data-mnc)
- **Original Size:** 2,000,000 rows, 12 columns
- **Features:** `Department`, `Job_Title`, `Hire_Date`, `Location`, `Performance_Rating`, `Experience_Years`, `Status`, `Work_Mode`, `Salary_INR`
- **Target Variable:** `Salary_INR` (converted to local currency `Salary_VND`)
- **Missing Values:** None

The dataset is programmatically downloaded using `kagglehub`. To download manually:
```bash
wget "https://www.kaggle.com/api/v1/datasets/download/rohitgrewal/hr-data-mnc"
unzip hr-data-mnc -d hr_data
```

## Project Structure

```text
annual-salary-prediction/
├── notebooks/                              # Experiment notebooks (ordered by pipeline stage)
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_train_linear_regression.ipynb
│   ├── 03_train_decision_tree.ipynb
│   ├── 04_train_random_forest.ipynb
│   ├── 05_train_xgboost.ipynb
│   ├── 06_final_train.ipynb
│   └── 07_predict_test.ipynb
├── src/                                    # Production source code
│   ├── train.py                            # Train & serialize final XGBoost model
│   ├── predict.py                          # FastAPI inference server
│   ├── test.py                             # Client test script for /predict endpoint
│   └── predict_test.py                     # Alternate test script (scratchpad)
├── models/                                 # Serialized model artifacts
│   └── ml_xgboost.bin                      # Trained XGBoost model + DictVectorizer
├── docs/                                   # Screenshots & documentation assets
│   ├── deploy_model_fly.io.png
│   ├── fastapi_docs_fly.io.png
│   └── try_it_out_done.png
├── Dockerfile                              # Container image definition
├── fly.toml                                # Fly.io deployment configuration
├── pyproject.toml                          # Project metadata & dependencies
├── uv.lock                                 # Locked dependency versions
├── .python-version                         # Python version pin for uv
└── README.md
```

## Data Preparation

The following steps were applied to clean and prepare the dataset:

1. **Sampling:** Reduced to 600,000 rows (30%) for faster training cycles.
2. **Feature Removal:** Dropped identifiers (`Employee_ID`, `Full_Name`) and `Hire_Date` (high correlation with `Experience_Years`).
3. **Feature Engineering:** Extracted country from `Location` to reduce cardinality; mapped `Performance_Rating` integers to string categories (`rating1`–`rating5`).
4. **Target Transformation:** Converted `Salary_INR` → `Salary_VND` (×296.77), applied `np.log1p()` to handle right-skewed distribution.
5. **Encoding:** Used `DictVectorizer` for one-hot encoding of categorical features.

*See `notebooks/01_eda_and_preprocessing.ipynb` for detailed EDA.*

## Model Training & Evaluation

Four models were evaluated with hyperparameter tuning:

| Model | Hyperparameters | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | Default | 0.287 | 0.498 | 1.3% |
| Decision Tree Regressor | `max_depth`=10, `max_leaf_nodes`=15, `min_samples_leaf`=4200 | 0.288 | 0.495 | 1.3% |
| Random Forest Regressor | `n_estimators`=45, `max_depth`=10, `max_features`=150 | 0.288 | 0.495 | 1.3% |
| **XGBoost (Final)** | `eta`=0.3, `max_depth`=10, `min_child_weight`=1, `num_boost_round`=81 | **0.289** | **0.491** | **1.3%** |

All models perform similarly. **XGBoost** was selected as the final model for its scalability and production robustness.

*See `notebooks/02–05` for tuning experiments.*

## Setup & Virtual Environment

Requires **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
pip install uv

# Reproduce the environment from lock file
uv sync --locked
```

> **Note:** To run `src/train.py` locally, you also need `kagglehub` and `pandas`:
> ```bash
> uv pip install kagglehub[pandas-datasets] pandas
> ```

## Deployment Workflow

### 1. Train & Save the Model
```bash
uv run python src/train.py
# → saves model to models/ml_xgboost.bin
```

### 2. Start the Inference API
```bash
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696 --reload
```

### 3. Test the Endpoint
```bash
# In a separate terminal
uv run python src/test.py
```

## Containerization (Docker)

```bash
# Build image
docker build -t predict-annual-salary .

# Run container
docker run -it --rm -p 9696:9696 predict-annual-salary
```

## Cloud Deployment (Fly.io)

```bash
# Authenticate & initialize
fly auth signup
fly launch --generate-name

# Deploy
fly deploy

# Cleanup after testing
fly apps destroy <app-name>
```

### Screenshots

![Fly.io Live Deployment](docs/deploy_model_fly.io.png)

![FastAPI Auto-Docs](docs/fastapi_docs_fly.io.png)

![Swagger UI Test Result](docs/try_it_out_done.png)