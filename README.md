# Churn Prediction MLOps on AWS

An end-to-end MLOps pipeline for customer churn prediction, featuring automated data validation, preprocessing, multi-model training with hyperparameter tuning, experiment tracking via MLflow, and AWS cloud infrastructure managed with Terraform.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [AWS Infrastructure](#aws-infrastructure)
- [Setup](#setup)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [MLflow Tracking](#mlflow-tracking)
- [Data Version Control](#data-version-control)

---

## Overview

This project builds a production-ready MLOps pipeline to predict customer churn using a banking dataset. It covers the full ML lifecycle:

- **Data validation** — schema, null, and range checks
- **Preprocessing** — encoding, scaling, and SMOTE oversampling
- **Training** — Logistic Regression, Random Forest, and Gradient Boosting with GridSearchCV
- **Experiment tracking** — MLflow hosted on an EC2 instance with S3 artifact storage
- **Infrastructure as code** — Terraform provisions all AWS resources (EC2, S3, SageMaker, IAM, ECR)

**Dataset**: [Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn) — 10,000 customer records with 18 features, target variable `Exited` (1 = churned).

---

## Architecture

```
Local Development / SageMaker Studio
           |
           v
  [Data Validation] ──── src/data/data_validation.py
           |
           v
  [Preprocessing] ─────── src/data/preprocessing.py
    - Drop non-predictive columns
    - One-hot encode categoricals
    - StandardScaler normalization
    - SMOTE oversampling
    - 80/20 stratified split
           |
           v
  [Model Training] ────── src/training/train.py
    - Logistic Regression (GridSearchCV)
    - Random Forest (GridSearchCV)
    - Gradient Boosting (GridSearchCV)
           |
           v
  [MLflow Tracking] ───── EC2 t3.micro (eu-north-1)
    - Parameters, metrics, artifacts
    - S3 artifact backend
           |
           v
  [S3 Buckets]
    - customer-churn-raw-data
    - customer-churn-processed-data
    - customer-churn-model-artifacts
```

---

## Project Structure

```
churn-prediction-mlops-aws/
├── data/
│   ├── raw/                        # Raw CSV (DVC-tracked)
│   └── processed/                  # Train/test splits (generated)
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # Data cleaning, encoding, SMOTE
│   │   └── data_validation.py      # Schema, null, range validation
│   ├── training/
│   │   └── train.py                # Multi-model training + MLflow logging
│   ├── evaluation/                 # (planned)
│   ├── monitoring/                 # (planned)
│   └── pipeline/                   # (planned - SageMaker Pipelines)
├── terraform/
│   ├── main.tf                     # Provider config & S3 remote backend
│   ├── variables.tf                # Region, IP address variables
│   ├── terraform.tfvars            # Variable values (gitignored)
│   ├── aws-ec2.tf                  # MLflow tracking server
│   ├── aws-s3.tf                   # Data and artifact buckets
│   ├── aws-iam.tf                  # SageMaker execution role
│   ├── aws-sagemaker.tf            # SageMaker Studio domain
│   └── aws-ecr.tf                  # Training & inference ECR repos
├── tests/
│   └── test_mlflow_logging.py      # MLflow connectivity smoke test
└── README.md
```

---

## AWS Infrastructure

All infrastructure is defined in [terraform/](terraform/) and deployed to `eu-north-1`.

| Resource | Type | Purpose |
|---|---|---|
| EC2 t3.micro | `aws-ec2.tf` | MLflow tracking server (port 5000) |
| S3 — raw data | `aws-s3.tf` | Raw CSV input, versioning enabled |
| S3 — processed data | `aws-s3.tf` | Preprocessed train/test splits |
| S3 — model artifacts | `aws-s3.tf` | MLflow artifact backend |
| S3 — terraform state | `main.tf` | Remote Terraform state backend |
| SageMaker Domain | `aws-sagemaker.tf` | Studio workspace (IAM auth) |
| IAM Role | `aws-iam.tf` | Scoped SageMaker execution role |
| ECR — training | `aws-ecr.tf` | Training job Docker images |
| ECR — inference | `aws-ecr.tf` | Inference/endpoint Docker images |

### Deploying Infrastructure

```bash
cd terraform

# Initialise (downloads providers, configures S3 backend)
terraform init

# Preview changes
terraform plan

# Apply
terraform apply
```

> **Note**: `terraform.tfvars` sets `aws_region = "eu-north-1"` and `local_ip_addr` to your public IP. The EC2 security group restricts MLflow (port 5000) and SSH (port 22) to this IP only.

---

## Setup

### Prerequisites

- Python 3.10+
- AWS CLI configured (`aws configure`)
- Terraform >= 1.0
- DVC (`pip install dvc[s3]`)

### Install Python dependencies

```bash
pip install -r requirements.txt
```

> Core dependencies: `scikit-learn`, `imbalanced-learn` (SMOTE), `mlflow`, `boto3`, `pandas`, `numpy`

### Pull raw data with DVC

```bash
dvc pull
```

---

## Usage

### 1. Validate raw data

```bash
python -c "from src.data.data_validation import validate_churn_csv; validate_churn_csv('data/raw/Customer-Churn-Records.csv')"
```

Runs three checks — schema, nulls, and value ranges — and prints a pass/fail report.

### 2. Preprocess data

```bash
python src/data/preprocessing.py \
  --input-path data/raw/Customer-Churn-Records.csv \
  --output-path data/processed/
```

Outputs `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` to `data/processed/`.

### 3. Train models

```bash
python src/training/train.py
```

Trains Logistic Regression, Random Forest, and Gradient Boosting with GridSearchCV. All runs are logged to MLflow under the experiment **"Churn Prediction Models"**.

Set the MLflow tracking URI before running if using the EC2 server:

```bash
export MLFLOW_TRACKING_URI=http://<EC2_PUBLIC_IP>:5000
python src/training/train.py
```

### 4. Smoke test MLflow connection

```bash
python tests/test_mlflow_logging.py
```

---

## ML Pipeline

### Models & Hyperparameter Grids

| Model | Hyperparameters Searched |
|---|---|
| Logistic Regression | `C`: [0.01, 0.1, 1, 10], `penalty`: l2 |
| Random Forest | `n_estimators`: [50, 100, 200], `max_depth`: [None, 10, 20], `min_samples_split`: [2, 5, 10] |
| Gradient Boosting | `n_estimators`: [50, 100, 200], `learning_rate`: [0.01, 0.1, 0.2], `max_depth`: [3, 5, 7] |

All models use **ROC AUC** as the cross-validation scoring metric.

### Preprocessing Steps

1. Drop `RowNumber`, `CustomerId`, `Surname`
2. One-hot encode `Geography`, `Gender`, `Card Type`
3. `StandardScaler` on `CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`, `Point Earned`
4. SMOTE on training set to address class imbalance (~20% churn rate)
5. Stratified 80/20 train/test split (random state 42)

### Metrics Logged

- `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`
- Full classification report and confusion matrix saved as MLflow artifacts

---

## MLflow Tracking

MLflow is hosted on an EC2 t3.micro instance with:
- **Backend store**: SQLite at `/mlflow/mlflow.db`
- **Artifact store**: `s3://customer-churn-model-artifacts-<account-id>/mlflow`

Access the UI at `http://<EC2_PUBLIC_IP>:5000` (IP-restricted to your machine).

---

## Data Version Control

Raw data is tracked with DVC. The `.dvc` file records the MD5 hash of `Customer-Churn-Records.csv` to ensure reproducibility.

```bash
# After updating raw data
dvc add data/raw/Customer-Churn-Records.csv
git add data/raw/Customer-Churn-Records.csv.dvc
git commit -m "update raw data"
dvc push
```
