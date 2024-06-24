import joblib
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import classification_report
from src.logger import logging

def load_model(model_path):
    return joblib.load(model_path)

def load_data(data_path):
    return pd.read_csv(data_path)

def predict(model, X):
    return model.predict(X)

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    # Load the model
    model_path = root_path / 'models' / 'models' / f'{sys.argv[1].lower()}.joblib'
    model = load_model(model_path)

    # Load the test data
    test_data_path = root_path / sys.argv[2]
    test_data = load_data(test_data_path)

    # Separate features and target
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']

    # Make predictions
    y_pred = predict(model, X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Save predictions
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_path = root_path / 'data' / 'predictions' / f'{sys.argv[1]}_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()