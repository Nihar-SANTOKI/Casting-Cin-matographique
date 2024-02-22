import os
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)
sys.path.append('..')

# Fonction pour évaluer le modèle avec le score R² et l'erreur absolue moyenne (MAE)
def evaluate_model_with_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae