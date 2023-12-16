import pandas as pd
import requests

# Charger le DataFrame crypto_df à partir du fichier CSV
crypto_df = pd.read_csv('https://raw.githubusercontent.com/NGALENAL1004/datasets/master/crypto_info.csv', index_col=0)

# Créer une liste pour stocker les crypto_codes à supprimer
codes_to_remove = []

# Endpoint pour la requête API
endpoint = 'https://min-api.cryptocompare.com/data/histoday'

# Récupérer les informations pour chaque crypto_code
for crypto_code in crypto_df.index:
    # Requête pour récupérer les données historiques
    res = requests.get(f'{endpoint}?fsym={crypto_code}&tsym=EUR&limit=2')
    data = res.json()
    
    # Vérification si 'Data' est vide ou 'time' n'est pas présent
    if not data['Data'] or 'time' not in data['Data'][0]:
        codes_to_remove.append(crypto_code)  # Ajouter le code à la liste des codes à supprimer

# Supprimer les lignes correspondant aux crypto_codes sans index 'time'
crypto_df = crypto_df.drop(codes_to_remove, axis=0)
# Sauvegarder les données filtrées dans un nouveau fichier CSV
crypto_df.to_csv('crypto_info_filtered.csv')


"""L'exécution de se scritpt peut prendre jusqu'à 40 minutes 
en fonction des capactiés de la machine"""