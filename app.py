import os
import joblib
import mlflow
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
from src.logger import logging

app = Flask(__name__)

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Match the URI in your hyperparameter tuning script
experiment_name = "Hyperparameter Tuning"

# Load the best model
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(experiment)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.best_f1_score DESC"])
        if len(runs) > 0:
            best_run = runs.iloc[0]
            model_uri = f"runs:/{best_run.run_id}/{best_run.data.tags['model_name']}_best"
            model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Loaded best model from run: {best_run.run_id}")
        else:
            raise Exception("No runs found in the MLflow experiment")
    else:
        raise Exception(f"MLflow experiment '{experiment_name}' not found")
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    # Fallback to loading from local file
    root_path = Path(__file__).parent
    model_path = root_path / 'models' / 'tuned_models' / 'randomforest_tuned.joblib'
    model = joblib.load(model_path)
    print(f"Loaded model from local file: {model_path}")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         features = pd.DataFrame(data['features'])
#         prediction = model.predict(features).tolist()
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/healthcheck', methods=['GET'])
# def healthcheck():
#     return jsonify({'status': 'healthy'}), 200

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)