import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib
from sklearn.metrics import confusion_matrix
from src.logger import logging

def load_data(file_path):
    return pd.read_csv(file_path)

def create_plot_directory(root_path, model_name):
    plot_dir = root_path / 'reports' / 'figures' / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def plot_confusion_matrix(y_true, y_pred, model_name, plot_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name}_confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, X, model_name, plot_dir):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(plot_dir / f'{model_name}_feature_importance.png')
        plt.close()

def plot_prediction_distribution(predictions, model_name, plot_dir):
    plt.figure(figsize=(12, 8))
    sns.countplot(x=predictions)
    plt.title(f'Distribution of Predictions - {model_name}')
    plt.xlabel('Predicted Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name}_prediction_distribution.png')
    plt.close()

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    # Load validation data
    val_path = root_path / 'data' / 'processed' / sys.argv[1]
    val_data = load_data(val_path)

    # Load test predictions
    submission_path = root_path / 'data' / 'submission' / 'submission.csv'
    test_predictions = load_data(submission_path)

    # Load model and label encoder
    model_name = 'randomforest'  # Change this if you're using a different model
    model_path = root_path / 'models' / 'models' / f'{model_name}.joblib'
    le_path = root_path / 'models' / 'transformers' / 'label_encoder.joblib'
    model = joblib.load(model_path)
    le = joblib.load(le_path)

    # Create plot directory
    plot_dir = create_plot_directory(root_path, model_name)

    # Prepare validation data
    X_val = val_data.drop(columns=['Target', 'id'], errors='ignore')
    y_val = val_data['Target']

    # Make predictions on validation data and inverse transform
    val_predictions = model.predict(X_val)
    val_predictions = le.inverse_transform(val_predictions)
    y_val = le.inverse_transform(y_val)

    # Plot confusion matrix
    plot_confusion_matrix(y_val, val_predictions, model_name, plot_dir)
    logging.info(f"Confusion matrix plot saved for {model_name}")

    # Plot feature importance
    plot_feature_importance(model, X_val, model_name, plot_dir)
    logging.info(f"Feature importance plot saved for {model_name}")

    # Plot prediction distribution for test set
    plot_prediction_distribution(test_predictions['Target'], model_name, plot_dir)
    logging.info(f"Prediction distribution plot saved for {model_name}")

    # Additional analysis: correlation heatmap of features
    plt.figure(figsize=(14, 12))
    sns.heatmap(X_val.corr(), cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name}_feature_correlation.png')
    plt.close()
    logging.info(f"Feature correlation heatmap saved for {model_name}")

    logging.info(f"All plots have been generated successfully and saved in {plot_dir}")

if __name__ == "__main__":
    main()