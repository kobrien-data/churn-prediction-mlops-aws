import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import argparse

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train a logistic regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_model_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a random forest model."""
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_model_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """Train a gradient boosting model."""
    model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the model using classification report, confusion matrix, and ROC AUC score."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)
    
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    return {
        "classification_report": class_report,
        "confusion_matrix": confusion_matrix,
        "roc_auc_score": auc_score
    }

def log_to_mlflow(model, model_name: str) -> None:
    """Log the model to MLflow."""
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")  # Update with EC2 MLflow server URI later
    mlflow.set_experiment("Churn Prediction Models")
    
    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("roc_auc_score", evaluate_model(model, X_test, y_test)["roc_auc_score"])

def run_experiment(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Run the experiment by training and evaluating multiple models."""

    lr_model = train_model_logistic_regression(X_train, y_train)
    rf_model = train_model_random_forest(X_train, y_train)
    gb_model = train_model_gradient_boosting(X_train, y_train)
    
    print("Logistic Regression Performance:")
    evaluate_model(lr_model, X_test, y_test)
    
    print("\nRandom Forest Performance:")
    evaluate_model(rf_model, X_test, y_test)
    
    print("\nGradient Boosting Performance:")
    evaluate_model(gb_model, X_test, y_test)
    
    log_to_mlflow(rf_model, "RandomForestClassifier")
    log_to_mlflow(gb_model, "GradientBoostingClassifier")
    log_to_mlflow(lr_model, "LogisticRegression")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='preprocessed_data')
    args = parser.parse_args()
    X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{args.data_dir}/y_train.csv").squeeze()  # Convert to Series
    X_test = pd.read_csv(f"{args.data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{args.data_dir}/y_test.csv").squeeze()  # Convert to Series
    
    
    run_experiment(X_train, y_train, X_test, y_test)