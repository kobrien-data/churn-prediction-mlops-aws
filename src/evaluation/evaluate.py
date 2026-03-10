import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import mlflow
import argparse

def load_model(file_path: str):
    """Load a trained model from a MLFlow."""
    return joblib.load(file_path)

def load_test_data(X_test_path: str, y_test_path: str): 
    """Load test data from CSV files."""
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    return X_test, y_test

def compute_classification_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics for the given model and test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc_score': roc_auc_score(y_test, y_proba)
    }

def compute_threshold_metrics(y_proba: pd.Series, y_test: pd.Series) -> dict:
    """Compute metrics at different thresholds for the given model and test data."""
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    metrics = {}
    
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        metrics[threshold] = {
            'classification_report': classification_report(y_test, y_pred_threshold),
            'confusion_matrix': confusion_matrix(y_test, y_pred_threshold)
        }
    
    return metrics

def plot_confusion_matrix(conf_matrix, model_name: str, output_path: str):
    """Plot the confusion matrix for the given model and saves it to MLFlow"""
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{output_path}{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_test: pd.Series, y_proba: pd.Series, model_name: str, output_path: str):
    """Plot the ROC curve for the given model and saves it to MLFlow"""
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'{output_path}{model_name}_roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_test: pd.Series, y_proba: pd.Series, model_name: str, output_path: str):
    """Plot the Precision-Recall curve for the given model and saves it to MLFlow"""
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(f'{output_path}{model_name}_precision_recall_curve.png')
    plt.close()

def plot_feature_importance(model, feature_names: list, model_name: str, output_path: str):
    """Plot the feature importance for the given model and saves it to MLFlow"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
        plt.title(f'Feature Importance for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.savefig(f'{output_path}{model_name}_feature_importance.png')
        plt.close()

def generate_evaluation_report(metrics_dict, model_name, output_path):
    """Generate a comprehensive evaluation report for the given model and saves it to MLFlow"""
    with open(f'{output_path}{model_name}_evaluation_report.txt', 'w') as f:
        f.write(f"Evaluation Report for {model_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Classification Report:\n")
        f.write(metrics_dict['classification_report'] + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(metrics_dict['confusion_matrix']) + "\n\n")
        
        f.write(f"ROC AUC Score: {metrics_dict['roc_auc_score']:.4f}\n")

def log_evaluation_results_to_mlflow(metrics, plots):
    """Log evaluation results to MLFlow."""
    with mlflow.start_run():
        mlflow.log_metrics({'roc_auc_score': metrics['roc_auc_score']})
        for plot in plots:
            mlflow.log_artifact(plot)

def compare_models(model_paths, X_test, y_test, output_path):
    """Compare multiple models based on their evaluation metrics."""
    results = {}
    
    for model_name, model_path in model_paths.items():
        model = load_model(model_path)
        metrics = compute_classification_metrics(model, X_test, y_test)
        results[model_name] = metrics
        
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, output_path=output_path)
        plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], model_name, output_path=output_path)
        plot_precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1], model_name, output_path=output_path)
        plot_feature_importance(model, feature_names=X_test.columns, model_name=model_name, output_path=output_path)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='src/data/models/best_model.pkl')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed_data/')
    parser.add_argument('--output-dir', type=str, default='src/data/evaluation_results/')
    args = parser.parse_args()

    X_test, y_test = load_test_data(args.data_dir + 'X_test.csv', args.data_dir + 'y_test.csv')
    model = load_model(args.model_path)
    metrics = compute_classification_metrics(model, X_test, y_test)
    plot_confusion_matrix(metrics['confusion_matrix'], 'BestModel', args.output_dir)
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], 'BestModel', args.output_dir)
    plot_precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1], 'BestModel', args.output_dir)
    plot_feature_importance(model, feature_names=X_test.columns, model_name='BestModel', output_path=args.output_dir)
    generate_evaluation_report(metrics, 'BestModel', args.output_dir)
    plots = [
        f'{args.output_dir}BestModel_confusion_matrix.png',
        f'{args.output_dir}BestModel_roc_curve.png',
        f'{args.output_dir}BestModel_precision_recall_curve.png',
        f'{args.output_dir}BestModel_evaluation_report.txt'
    ]
    if hasattr(model, 'feature_importances_'):
        plots.append(f'{args.output_dir}BestModel_feature_importance.png')
    log_evaluation_results_to_mlflow(metrics, plots)