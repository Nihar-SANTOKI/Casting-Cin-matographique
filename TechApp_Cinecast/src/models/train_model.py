import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)
sys.path.append('..')

#Entrainment des modéles
# Fonction pour la Régression Linéaire
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Fonction pour entraîner les Arbres de Décision avec différentes profondeurs
def train_decision_tree_with_depths(X_train, y_train, max_depths):
    models = []
    for depth in max_depths:
        model_dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model_dt.fit(X_train, y_train)
        models.append(model_dt)
    return models

# Fonction pour entraîner les Forêts Aléatoires avec différents n_estimators et profondeurs
def train_random_forest_with_params(X_train, y_train, n_estimators, max_depths):
    models = []
    for n_est in n_estimators:
        for depth in max_depths:
            model_rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42)
            model_rf.fit(X_train, y_train)
            models.append(model_rf)
    return models

# Fonction pour entraîner les K-Nearest Neighbors (KNN) avec différents n_neighbors
def train_knn_with_neighbors(X_train, y_train, n_neighbors):
    models = []
    for neighbor in n_neighbors:
        model = KNeighborsRegressor(n_neighbors=neighbor)
        model.fit(X_train, y_train)
        models.append(model)
    return models

def train_multi_output_regression(X_train, y_train):
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)
    return model