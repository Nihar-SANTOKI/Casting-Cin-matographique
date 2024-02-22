import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)
sys.path.append('..')

# Fonction pour prédire en fonction des entrées de l'utilisateur
def predict_movie(model):
    # Entrées de l'utilisateur
    user_budget = float(input("Entrez votre budget : "))
    user_genre = input("Entrez le genre : ")
    num_stars = int(input("Entrez le nombre d'étoiles (1, 2 ou 3) : "))

    # Mise à l'échelle de l'entrée utilisateur pour le budget
    scaler = StandardScaler()
    user_budget_scaled = scaler.fit_transform([[user_budget]])[0][0]

    # Codage du genre utilisateur
    label_encoder = LabelEncoder()
    user_genre_encoded = label_encoder.fit_transform([user_genre])[0]

    # Préparation des caractéristiques d'entrée pour la prédiction
    user_input = pd.DataFrame([[user_genre_encoded, user_budget_scaled, 0, 0, 0]], columns=['genre', 'budget', 'star', 'star2', 'star3'])

    # Prédiction en fonction du nombre d'étoiles entré par l'utilisateur
    if num_stars == 1:
        prediction = model.predict(user_input[['genre', 'budget', 'star']])
    elif num_stars == 2:
        prediction = model.predict(user_input[['genre', 'budget', 'star', 'star2']])
    elif num_stars == 3:
        prediction = model.predict(user_input[['genre', 'budget', 'star', 'star2', 'star3']])
    else:
        raise ValueError("Le nombre d'étoiles doit être 1, 2 ou 3")

    return prediction

def predict_movie_attributes(model, budget, genre, num_stars):
    # Création d'un DataFrame pour contenir les entrées de l'utilisateur
    user_input = pd.DataFrame({'budget': [budget], 'genre': [genre], 'num_stars': [num_stars]})
    
    # Effectuer un encodage one-hot pour 'genre'
    user_input = perform_one_hot_encoding(user_input, 'genre')
    
    # Créer X_test avec les caractéristiques des entrées de l'utilisateur
    colonnes_pour_X = ['budget'] + [colonne for colonne in user_input.columns if colonne.startswith('genre_')]
    X_test = user_input[colonnes_pour_X]
    
    # Prédire en utilisant le modèle fourni
    predictions = model.predict(X_test)
    
    # Extraire les prédictions pour recette, star, star2 et star3
    recette_prediction, star_prediction, star2_prediction, star3_prediction = predictions[0]

    return {
        'recette_prediction': recette_prediction,
        'star_prediction': star_prediction,
        'star2_prediction': star2_prediction,
        'star3_prediction': star3_prediction
    }