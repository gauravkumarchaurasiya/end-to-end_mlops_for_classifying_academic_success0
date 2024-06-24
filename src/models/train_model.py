import joblib
import sys
import logging
import pandas as pd
from yaml import safe_load
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from src.logger import logging
from src.models.models_list import models

TARGET = 'Target'

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a CSV file."""
    return pd.read_csv(path)

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    """Split a DataFrame into feature matrix X and target vector y."""
    X = dataframe.drop(columns=target_column)
    y = dataframe[target_column]
    return X, y

def train_model(model, X_train, y_train):
    """Train a machine learning model."""
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, f1, precision, recall

def save_model(model, save_path: Path):
    """Save a trained model to a file."""
    joblib.dump(value=model, filename=save_path)

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    training_data_path = root_path /'data'/'processed'/ sys.argv[1]

    train_data = load_dataframe(training_data_path)
    X_train, y_train = make_X_y(dataframe=train_data, target_column=TARGET)
    logging.info("Train Test split successful")

    with open('params.yaml') as f:
        params = safe_load(f)

    logging.info(f"Models to be trained: {models}")

    model_names = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for model_name, model in models.items():
        logging.info(f'{model_name} is training...')
        trained_model = train_model(model=model, X_train=X_train, y_train=y_train)

        accuracy, f1, precision, recall = evaluate_model(model=trained_model, X_test=X_train, y_test=y_train)
        logging.info(f"{model_name} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

        model_names.append(model_name)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        model_output_path = root_path / 'models' / 'models'
        model_output_path.mkdir(exist_ok=True)
        model_output_path_ = model_output_path / f'{model_name.lower()}.joblib'
        save_model(model=trained_model, save_path=model_output_path_)

    logging.info("Step 4: Completed Model Training")

if __name__ == "__main__":
    main()