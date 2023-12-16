import pandas as pd
import requests
import json
import os

# Vérifier si le fichier CSV existe
if not os.path.exists('crypto_info.csv'):
    # Si le fichier n'existe pas, télécharger les données depuis l'API
    url = 'https://min-api.cryptocompare.com/data/all/coinlist'
    response = requests.get(url)
    data = response.json()
    crypto_dict = data['Data']

    # Créer une DataFrame avec les données récupérées
    crypto_df = pd.DataFrame(crypto_dict).T
    crypto_df.to_csv('crypto_info.csv')
else:
    # Si le fichier existe, lire les données depuis le fichier CSV
    crypto_df = pd.read_csv('crypto_info.csv', index_col=0)


