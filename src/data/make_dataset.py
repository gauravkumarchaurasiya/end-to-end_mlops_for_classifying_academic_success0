import sys
import logging
from pathlib import Path
import pandas as pd
from yaml import safe_load
from src.logger import logging
from sklearn.model_selection import train_test_split

def load_raw_data(input_path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows,columns = raw_data.shape
    logging.info(f"{input_path.stem} data read having {rows} rows and {columns} columns")
    return raw_data

def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
            
    except FileNotFoundError as e:
        logging.error('Parameters file not found, Switching to default value of train test split ...')
        default_dict = {'test_size': 0.2,'random_state':None}
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return test_size, random_state
    else:
        logging.info("Parameters file read successfully")
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return test_size, random_state
    
def train_val_split(data:pd.DataFrame,test_size:float,random_state:int)-> tuple[pd.DataFrame,pd.DataFrame]:
    train_data,val_data = train_test_split(data,test_size=test_size,random_state=random_state)
    logging.info(f"The Parameter values are {test_size} for test size and {random_state} for random state")
    logging.info(f"Data is split into train test split with shape{train_data.shape} and val split with shape {val_data.shape}")
    return train_data,val_data

def save_data(data:pd.DataFrame,output_path:Path):
    data.to_csv(output_path,index=False)
    logging.info(f"{output_path.stem + output_path.suffix} data saved successfully to the {output_path} folder")
def main():
    # read the input file name from command
    input_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # interim data directory path 
    interim_data_path = root_path/'data'/'interim'
    interim_data_path.mkdir(parents=True,exist_ok=True)
    
    raw_df_path = root_path/'data'/'raw'/'extracted'/input_file_name
    raw_test_df_path = root_path/'data'/'raw'/'extracted'/test_file_name
    
    raw_df = load_raw_data(input_path=raw_df_path)
    raw_test_df = load_raw_data(input_path=raw_test_df_path)
    
    test_size,random_state = read_params('params.yaml')
    
    train_df,val_df = train_val_split(data = raw_df,test_size=test_size,random_state=random_state)
    
    save_data(data = train_df,output_path = interim_data_path/'train.csv')
    save_data(data = val_df,output_path = interim_data_path/'val.csv')
    save_data(data = raw_test_df,output_path = interim_data_path/'test.csv')
    logging.info(f"Step 2 : Completed DataFrame Creation[train.csv,val.csv,test.csv]")
    
if __name__=='__main__':
    main()