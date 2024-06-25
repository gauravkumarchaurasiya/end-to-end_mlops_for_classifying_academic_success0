import time
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import logging

# Import your existing functions
from train_model import load_dataframe, make_X_y, save_model

TARGET = 'Target'

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")  # Set to your MLflow tracking server
    experiment_name = "Hyperparameter Tuning"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # If the experiment doesn't exist, create a new one
            experiment_id = mlflow.create_experiment(experiment_name)
            logging.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        elif experiment.lifecycle_stage == 'deleted':
            # If the experiment is deleted, create a new one with a different name
            new_experiment_name = f"{experiment_name}_{int(time.time())}"
            experiment_id = mlflow.create_experiment(new_experiment_name)
            logging.info(f"Created new experiment '{new_experiment_name}' with ID: {experiment_id}")
        else:
            # If the experiment exists and is active, use it
            experiment_id = experiment.experiment_id
            logging.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
        
        mlflow.set_experiment(experiment_id)
    except mlflow.MlflowException as e:
        logging.error(f"Error setting up MLflow experiment: {e}")
        raise

    logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logging.info(f"MLflow experiment: {mlflow.get_experiment(experiment_id).name}")

def hyperparameter_tuning(model, param_dist, X_train, y_train, X_val, y_val, n_iter=100, cv=5):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, 
                                       scoring='f1_weighted', n_jobs=-1, verbose=2, random_state=42)
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    
    return best_model, random_search.best_params_, accuracy, f1, precision, recall

def main():
    setup_mlflow()
    
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    
    train_data_path = root_path / 'data' / 'processed' / 'train.csv'
    val_data_path = root_path / 'data' / 'processed' / 'val.csv'
    
    train_data = load_dataframe(train_data_path)
    val_data = load_dataframe(val_data_path)
    
    X_train, y_train = make_X_y(train_data, TARGET)
    X_val, y_val = make_X_y(val_data, TARGET)
    
    models_to_tune = {
        'RandomForest': (RandomForestClassifier(), {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }),
        'GradientBoosting': (GradientBoostingClassifier(), {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        })
    }
    
    for model_name, (model, param_dist) in models_to_tune.items():
        with mlflow.start_run(run_name=f"{model_name}_tuning"):
            logging.info(f"Starting hyperparameter tuning for {model_name}")
            
            best_model, best_params, accuracy, f1, precision, recall = hyperparameter_tuning(
                model, param_dist, X_train, y_train, X_val, y_val
            )
            
            logging.info(f"Best parameters for {model_name}: {best_params}")
            logging.info(f"Validation metrics - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
            
            # Log best parameters and final metrics
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_accuracy", accuracy)
            mlflow.log_metric("best_f1_score", f1)
            mlflow.log_metric("best_precision", precision)
            mlflow.log_metric("best_recall", recall)
            
            logging.info(f"Logged best parameters: {best_params}")
            logging.info(f"Logged best metrics: accuracy={accuracy}, f1_score={f1}, precision={precision}, recall={recall}")
            
            # Log the model
            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(best_model, f"{model_name}_best", signature=signature)
            
            # Save the model locally
            model_output_path = root_path / 'models' / 'tuned_models'
            model_output_path.mkdir(exist_ok=True)
            save_model(best_model, model_output_path / f'{model_name.lower()}_tuned.joblib')

if __name__ == "__main__":
    main()
