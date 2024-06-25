from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT='-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    encodings = ['utf-8', 'utf-16', 'utf-32', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file_obj:
                requirements = file_obj.read().splitlines()
            break  # If successful, exit the loop
        except UnicodeDecodeError:
            continue  # Try the next encoding
    
    if not requirements:
        raise ValueError(f"Unable to read {file_path} with any of the attempted encodings")
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements
setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project demonstrates the implementation of a robust MLOps pipeline to classify academic success using machine learning techniques. By leveraging an academic success dataset, we explore data preprocessing, model training, hyperparameter tuning, and deployment within a streamlined MLOps framework. The project showcases end-to-end automation, ensuring efficient model development, continuous integration, continuous deployment, and monitoring to maintain high model performance in predicting student success outcomes.',
    author='Gaurav Kumar Chaurasiya',
    license='',
    install_requires=get_requirements('requirements.txt')
)
