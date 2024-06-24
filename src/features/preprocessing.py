import sys
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from src.logger import logging

# Define the target column
TARGET = 'Target'

def save_transformer(path: Path, transformer):
    joblib.dump(transformer, path)
    logging.info(f"Transformer saved at {path}")

def read_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_dataframe(dataframe: pd.DataFrame, save_path: Path):
    dataframe.to_csv(save_path, index=False)
    logging.info(f"Data saved at {save_path}")

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    input_path = root_path / 'data' / 'interim'
    transformers_path = root_path / 'models' / 'transformers'
    processed_data_path = root_path / 'data' / 'processed'

    transformers_path.mkdir(parents=True, exist_ok=True)
    processed_data_path.mkdir(parents=True, exist_ok=True)

    for filename in sys.argv[1:]:
        file_path = input_path / filename
        df = read_dataframe(file_path)

        if filename == 'train.csv':
            X = df.drop(columns=[TARGET, 'id'])
            y = df[TARGET]

            course_column = ['Course']
            numerical_columns = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
                                 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                                 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                                 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
                                 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                                 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                                 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
                                 'Unemployment rate', 'Inflation rate', 'GDP']
            categorical_columns = [col for col in X.columns if col not in numerical_columns + course_column]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns),
                    ('course', OneHotEncoder(handle_unknown='ignore'), course_column)
                ],
                remainder='passthrough'
            )

            X_preprocessed = preprocessor.fit_transform(X)

            feature_names = (numerical_columns +
                             list(preprocessor.named_transformers_['course'].get_feature_names_out(course_column)) +
                             categorical_columns)

            X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
           
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_encoded_df = pd.DataFrame(y_encoded, columns=[TARGET])

            processed_df = pd.concat([X_preprocessed_df, y_encoded_df], axis=1)

            save_transformer(transformers_path / 'preprocessor.joblib', preprocessor)
            save_transformer(transformers_path / 'label_encoder.joblib', le)

            save_dataframe(processed_df, processed_data_path / filename)

        elif filename in ['val.csv', 'test.csv']:
            X = df.drop(columns=[TARGET, 'id'], errors='ignore')
            
            preprocessor = joblib.load(transformers_path / 'preprocessor.joblib')
            le = joblib.load(transformers_path / 'label_encoder.joblib')

            X_trans = preprocessor.transform(X)

            feature_names = (numerical_columns +
                             list(preprocessor.named_transformers_['course'].get_feature_names_out(['Course'])) +
                             categorical_columns)

            X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

            if TARGET in df.columns:
                y = df[TARGET]
                y_encoded = le.transform(y)
                y_encoded_df = pd.DataFrame(y_encoded, columns=[TARGET])
                transformed_df = pd.concat([X_trans_df, y_encoded_df], axis=1)
            else:
                transformed_df = X_trans_df

            save_dataframe(transformed_df, processed_data_path / filename)
    
    logging.info("Step 3: Completed Data Preprocessing")

if __name__ == "__main__":
    main()