import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.models.models_list import models
from src.logger import logging

TARGET = 'Target'
model_name = 'randomforest.joblib'  # You can modify this to use other models

def load_dataframe(path):
    """Load a DataFrame from a CSV file."""
    return pd.read_csv(path)

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    """Split a DataFrame into feature matrix X and target vector y."""
    df_copy = dataframe.copy()
    X = df_copy.drop(columns=[target_column, 'id'], errors='ignore')
    y = df_copy[target_column]
    return X, y

def get_predictions(model, X: pd.DataFrame):
    """Get predictions on data."""
    return model.predict(X)

def calculate_metrics(y_actual, y_predicted):
    """Calculate various classification metrics."""
    accuracy = accuracy_score(y_actual, y_predicted)
    f1 = f1_score(y_actual, y_predicted, average='weighted')
    precision = precision_score(y_actual, y_predicted, average='weighted')
    recall = recall_score(y_actual, y_predicted, average='weighted')
    return accuracy, f1, precision, recall

def evaluate_and_log(model, X, y, dataset_name):
    """Evaluate model on given data and log results."""
    y_pred = get_predictions(model, X)
    accuracy, f1, precision, recall = calculate_metrics(y, y_pred)
    
    print(f'\nMetrics for {dataset_name} dataset:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    logging.info(f'\nMetrics for {dataset_name} dataset:')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    
    # Load the model
    model_path = root_path / 'models' / 'models' / model_name
    model = joblib.load(model_path)
    
    # Load the LabelEncoder
    le_path = root_path / 'models' / 'transformers' / 'label_encoder.joblib'
    le = joblib.load(le_path)
    
    # Evaluate on validation set
    val_path = root_path / 'data' / 'processed' / sys.argv[1]
    val_data = load_dataframe(val_path)
    X_val, y_val = make_X_y(val_data, TARGET)
    evaluate_and_log(model, X_val, y_val, "Validation")
    
    # Make predictions on test set (no labels)
    test_path = root_path / 'data' / 'processed' / sys.argv[2]
    test_data = load_dataframe(test_path)
    X_test = test_data.drop(columns=['id'], errors='ignore')
    raw_test = load_dataframe(root_path/'data'/'interim'/'test.csv')
    predictions = get_predictions(model, X_test)
    
    # Convert predictions back to original categories
    predictions_original = le.inverse_transform(predictions)
    
    # Create the submission DataFrame
    submission_df = pd.DataFrame({'id': raw_test['id'], 'Target': predictions_original})
    
    # Create submission directory if it doesn't exist
    submission_path = root_path / 'data' / 'submission'
    submission_path.mkdir(exist_ok=True)
    
    # Save the submission file to a CSV
    submission_file = submission_path / 'submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file created successfully at {submission_file}")
    logging.info(f"Submission file created successfully at {submission_file}")

if __name__ == "__main__":
    main()