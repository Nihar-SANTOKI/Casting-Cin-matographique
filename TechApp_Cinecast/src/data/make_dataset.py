# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)
sys.path.append('..')

def load_data(file_path):
    """
    Chargement de dataset.
    Retourne une DataFrame.
    """
    return pd.read_csv(file_path)

