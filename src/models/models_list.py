from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from yaml import safe_load

with open('params.yaml') as f:
        params = safe_load(f)

models = {
    'RandomForest': RandomForestClassifier(**params.get('train_model', {}).get('random_forest', {})),
    'LogisticRegression': LogisticRegression(**params.get('train_model', {}).get('logistic_regression', {})),
    'SVC': SVC(**params.get('train_model', {}).get('svc', {})),
    'DecisionTree': DecisionTreeClassifier(**params.get('train_model', {}).get('decision_tree', {})),
    'GradientBoosting': GradientBoostingClassifier(**params.get('train_model', {}).get('gradient_boosting', {})),
    'AdaBoost': AdaBoostClassifier(**params.get('train_model', {}).get('adaboost', {})),
    'KNN': KNeighborsClassifier(**params.get('train_model', {}).get('knn', {})),
    'GaussianNB': GaussianNB(**params.get('train_model', {}).get('gaussian_nb', {}))
}