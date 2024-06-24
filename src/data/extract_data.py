import logging
import sys
from pathlib import Path
from zipfile import ZipFile
from src.logger import logging

def extract_zip_file(input_path:Path,output_path:Path):
    with ZipFile(file=input_path) as f:
        f.extractall(path=output_path)
        input_file_name = input_path.stem + input_path.suffix
        logging.info(f'{input_file_name} extracted successfully at the target path')
    
def main():
    
    current_path = Path(__file__)
    zip_file = sys.argv[1]
    root_path = current_path.parent.parent.parent
    raw_data_path = root_path/'data'/'raw'
    output_path = raw_data_path / 'Extracted'
    output_path.mkdir(parents=True,exist_ok=True)
    input_path = raw_data_path/'Zipped'
    
    extract_zip_file(input_path=input_path/zip_file,output_path=output_path)
    logging.info(f"Step 1 : Completed Data Extraction from Zip")
    
if __name__=='__main__':
    main()