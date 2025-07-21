import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

def train_models(X, y, task_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    if task_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB()
        }
        params = {
            "Logistic Regression": {"C": [0.1, 1, 10]},
            "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            "KNN": {"n_neighbors": [3, 5, 7]},
            "Decision Tree": {"max_depth": [None, 5, 10]},
            "Random Forest": {"n_estimators": [50, 100]}
        }
    else:  
        # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "SVR": SVR(),
            # "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }
        params = {
            "SVR": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            # "KNN Regressor": {"n_neighbors": [3, 5, 7]},
            "Decision Tree Regressor": {"max_depth": [None, 5, 10]},
            "Random Forest Regressor": {"n_estimators": [50, 100]}
        }

    trained_models = {}
    for name,model in models.items():
        try:
            grid=GridSearchCV(model, params.get(name, {}),cv=5,error_score="raise")
            grid.fit(X_train,y_train)
            trained_models[name]=grid.best_estimator_
        except Exception as e:
            print(f"{name} failed:{e}")

    return trained_models,X_test,y_test
