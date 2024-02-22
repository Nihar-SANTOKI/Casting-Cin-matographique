import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_directory)

from data.make_dataset import load_data
from features.build_features import split_data, scale_features, perform_label_encoding
from models.train_model import train_linear_regression, train_decision_tree_with_depths, train_random_forest_with_params, train_knn_with_neighbors
from models.predict_model import predict_movie
from evaluation.evaluation import evaluate_model_with_metrics

os.system('python data/make_dataset.py')
os.system('python features/build_features.py')
os.system('python models/predict_model.py')
os.system('python models/train_model.py')
os.system('python evaluation/evaluation.py')

# Main utilisant d'encodage Label_encoder + Multi output classification et prediction (Ne marche pas avec 1 ou 2 nombre d'acteur, uniquement 3)
def main():
    # Chargement des données à partir du fichier 'movie_dataset.csv'
    data = load_data('movie_dataset.csv')
    # Encodage des données catégoriques dans le jeu de données
    data_encoded = perform_label_encoding(data)

    # Séparation des données en variables explicatives (X) et cible (y)
    X = data_encoded[['genre', 'budget', 'star', 'star2', 'star3']]
    y = data_encoded[['gross', 'star', 'star2', 'star3']]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Mise à l'échelle des caractéristiques
    scaled_X_train, scaled_X_test = scale_features(X_train, X_test)
    
    # Entraînement et évaluation de la régression linéaire sans mise à l'échelle
    model_lr = train_linear_regression(X_train, y_train)
    mse_lr, r2_lr, mae_lr = evaluate_model_with_metrics(model_lr, X_test, y_test)
    print("Régression linéaire (sans mise à l'échelle) - Erreur quadratique moyenne:", mse_lr)
    print("Régression linéaire (sans mise à l'échelle) - Score R²:", r2_lr)
    print("Régression linéaire (sans mise à l'échelle) - Erreur absolue moyenne:", mae_lr, "\n")

    # Entraînement et évaluation de la régression linéaire avec mise à l'échelle
    model_lr_scaled = train_linear_regression(scaled_X_train, y_train)
    mse_lr_scaled, r2_lr_scaled, mae_lr_scaled = evaluate_model_with_metrics(model_lr_scaled, scaled_X_test, y_test)
    print("Régression linéaire (avec mise à l'échelle) - Erreur quadratique moyenne:", mse_lr_scaled)
    print("Régression linéaire (avec mise à l'échelle) - Score R²:", r2_lr_scaled)
    print("Régression linéaire (avec mise à l'échelle) - Erreur absolue moyenne:", mae_lr_scaled, "\n")
    
    # Entraînement et évaluation des arbres de décision avec différentes profondeurs
    tree_depths = [3, 5, 7]
    models_dt_depths = train_decision_tree_with_depths(X_train, y_train, tree_depths)
    models_dt_depths_scaled = train_decision_tree_with_depths(scaled_X_train, y_train, tree_depths)
    
    # Entraînement et évaluation des forêts aléatoires avec différents n_estimators et profondeurs
    tree_estimators = [50, 100, 250, 400, 500] 
    models_rf_params = train_random_forest_with_params(X_train, y_train, tree_estimators, tree_depths)
    models_rf_params_scaled = train_random_forest_with_params(scaled_X_train, y_train, tree_estimators, tree_depths)
    
    # Entraînement et évaluation des K plus proches voisins (KNN) avec différents n_neighbors
    knn_neighbors = [1, 2, 3, 4, 5]
    models_knn_neighbors = train_knn_with_neighbors(X_train, y_train, knn_neighbors)
    models_knn_neighbors_scaled = train_knn_with_neighbors(scaled_X_train, y_train, knn_neighbors)
    
    # Évaluation de tous les modèles
    models = [
        ("Arbres de décision avec différentes profondeurs", models_dt_depths),
        ("Forêts aléatoires avec différents n_estimators et profondeurs", models_rf_params),
        ("K plus proches voisins (KNN) avec différents n_neighbors", models_knn_neighbors)
    ]
                                                           
    models2 = [
        ("Arbres de décision avec différentes profondeurs - Scaled", models_dt_depths_scaled),
        ("Forêts aléatoires avec différents n_estimators et profondeurs - Scaled", models_rf_params_scaled),
        ("K plus proches voisins (KNN) avec différents n_neighbors - Scaled", models_knn_neighbors_scaled)
    ]

    for model_name, model_list in models:
        print(f"Évaluation de {model_name}:")
        for model in model_list:
            mse, r2, mae = evaluate_model_with_metrics(model, X_test, y_test)
            print(f"Modèle: {model} - Erreur quadratique moyenne: {mse}")
            print(f"Modèle: {model} - Score R²: {r2}")
            print(f"Modèle: {model} - Erreur absolue moyenne: {mae}\n")
    
    for model_name, model_list in models2:
        print(f"Évaluation de {model_name}:")
        for model in model_list:
            mse, r2, mae = evaluate_model_with_metrics(model, X_test, y_test)
            print(f"Modèle: {model} - Erreur quadratique moyenne: {mse}")
            print(f"Modèle: {model} - Score R²: {r2}")
            print(f"Modèle: {model} - Erreur absolue moyenne: {mae}\n")
    
    # Vous pouvez nottament essayer d'autres modéles
    # Exemple d'utilisation du dernier modèle KNN pour la prédiction
    prediction = predict_movie(models_rf_params[-1])
    print("Gross prédite:", prediction[0][0])
    print("Star prédite:", prediction[0][1])
    print("Star2 prédite:", prediction[0][2])
    print("Star3 prédite:", prediction[0][3])

if __name__ == "__main__":
    main()