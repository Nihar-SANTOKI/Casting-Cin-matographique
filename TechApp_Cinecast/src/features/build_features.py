import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)
sys.path.append('..')

# Fonction pour le codage des étiquettes
def perform_label_encoding(data):
    label_encoder = LabelEncoder()
    data['genre'] = label_encoder.fit_transform(data['genre'])
    data['star'] = label_encoder.fit_transform(data['star'])
    data['star2'] = label_encoder.fit_transform(data['star2'])
    data['star3'] = label_encoder.fit_transform(data['star3'])
    return data

def perform_one_hot_encoding(data, column):
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_cols = pd.DataFrame(encoder.fit_transform(data[[column]]))
    encoded_cols.columns = encoder.get_feature_names([column])
    data = pd.concat([data, encoded_cols], axis=1)
    data.drop(columns=[column], inplace=True)
    return data

# Fonction pour diviser les données
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Fonction pour mettre à l'échelle les caractéristiques
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test