import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import datetime as dt
from datetime import timedelta

### Titre de l'application
st.title("🪙CryptoForetell🪙: Prédiction de la valeur des cryptomonnaies")

## Initialisation de la date d'aujourd'hui
today = dt.date.today()

### Spécification des paramètres de l'action
st.sidebar.header("Sélectionnez la cryptomonnaie")

## Choix de la cryptomonnaie
# Récupération des données de cryptomonaies via le csv
crypto_df = pd.read_csv('https://raw.githubusercontent.com/NGALENAL1004/datasets/master/crypto_info_filtered.csv', index_col=0)
# Afficher une selectbox avec les cryptomonnaies disponibles
selected_crypto = st.sidebar.selectbox("Sélectionnez une cryptomonnaie", crypto_df['FullName'].tolist())
# Récupérer le code de la cryptomonnaie sélectionnée à partir du DataFrame
crypto_code = crypto_df[crypto_df['FullName'] == selected_crypto].index[0]

## Sélection des dates
# Affichage du calendrier pour la sélection de la date de fin (max_value = date actuelle)
end_date = st.sidebar.date_input("Entrez la date de fin (YYYY-MM-JJ):", today, max_value=today)
# Limite inférieur pour la date de début 
min_start_date = end_date - dt.timedelta(days=2000)
#min_start_date = '2018-06-01'
#min_start_date = dt.datetime.strptime(min_start_date, '%Y-%m-%d').date()
# Limite inférieur pour la date de début
max_start_date = end_date - dt.timedelta(days=100)
#max_start_date = dt.datetime.strptime(min_start_date, '%Y-%m-%d').date()
# Affichage du calendrier pour la sélection de la date de début
start_date = st.sidebar.date_input("Entrez la date de début (YYYY-MM-JJ):", min_start_date, min_value=min_start_date, max_value=max_start_date)
# Conversion de la date sélectionnée en format string
start_date_str = start_date.strftime('%Y-%m-%d')
# Conversion des dates en timestamps Unix
end_timestamp = int(dt.datetime.strptime(str(end_date), '%Y-%m-%d').timestamp())
# Calcul de la différence de jours entre la date de fin et la date de début
difference = end_date - start_date
# Récupération du nombre de jours à partir de la différence
limit = difference.days


### Récupération des données
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
## Requête avec les dates spécifiques (Nous utiliserons les données du Bitcoin)
res = requests.get(f'{endpoint}?fsym={crypto_code}&tsym=EUR&limit={limit}&toTs={end_timestamp}')
df = pd.DataFrame(json.loads(res.content)['Data'])
df = df.set_index('time')
df.index = pd.to_datetime(df.index, unit='s')
df = df[df.index >= start_date_str]

# Modification du format de l'index des dates
df.index = df.index.strftime('%d-%m-%Y')

###Description des données
st.subheader("Données de {} à {}".format(start_date, end_date))
# Suppression des colonnes inutiles
df = df.drop(['conversionType', 'conversionSymbol'], axis=1)
st.write(df.describe())

###Visualisation des prix de clôture avec les dates sur l'axe des abscisses
st.subheader("Cours du prix de cloture du {} à {}".format(start_date,end_date))
fig=plt.figure(figsize=(10, 6))
plt.plot(df.index, df['close'], label='Prix de clôture')
plt.title('Évolution des prix de clôture')
plt.xlabel('Date')
plt.ylabel('Prix de clôture (EUR)')
plt.grid(True)
#Formater les dates sur l'axe x
plt.xticks(df.index[::50], rotation=45)  # Récupère toutes les 50 dates pour éviter la surcharge
plt.tight_layout()  # Ajustement pour éviter que les dates se chevauchent
st.pyplot(fig)

###Visualisation de la courbe de la moyenne mobile à 10 et 30 jours et du prix de clôture en fonction du temps
st.subheader("Évolution de la moyenne mobile à 10 et 30 jours, puis du prix de cloture au cours du temps")
ma10 = df.close.rolling(10).mean()
ma30 = df.close.rolling(30).mean()
fig2=plt.figure(figsize=(15,6))
plt.plot(df.close)
plt.plot(ma10, 'r')
plt.plot(ma30,'g')
plt.title('Évolution de la moyenne mobile à 10 et 30 jours, puis du prix de cloture au cours du temps')
plt.xticks(df.index[::50], rotation=45)  # Récupère toutes les 50 dates pour éviter la surcharge
plt.xlabel('Date')
plt.ylabel('Prix de clôture (EUR)')
# Afficher la grille si nécessaire
plt.grid(True)
st.pyplot(fig2)

###Calcul de l'index pour diviser les données (70% pour l'entraînement, 30% pour le test)
train_size = 0.7
split_index = int(len(df) * train_size)
# Séparation des données en ensembles d'entraînement et de test
df_training = pd.DataFrame(df['close'][:split_index])  # Données d'entraînement (70%)
df_testing = pd.DataFrame(df['close'][split_index:])   # Données de test (30%)

###Scaling de la Traing Data et transformation en matrice
scaler = MinMaxScaler(feature_range=(0,1))
df_training_array = scaler.fit_transform(df_training)

###Division de x_train et y_train par pas de 10 jours
x_train = []
y_train = []
for i in range(10, df_training_array.shape[0]):
  x_train.append(df_training_array[i-10: i ])
  y_train.append(df_training_array[i, 0])
#Conversion de x_train et y_train en matrices
x_train, y_train = np.array(x_train), np.array(y_train)

###Chargement du model
model=load_model('cryptoforetellmodel.h5')

###Préparation des données de test
#Charger les 10 dernières lignes du Training Set pour les ajouter au (début) Testing Set
past_10_days = df_training.tail(10)
#Ajouter les 10 dernières lignes aux Data Frame final du Testing
df_testing = pd.concat([past_10_days, df_testing])
#Scaling pour changer l'échelle du Testing Set à [0-1]
df_testing_array = scaler.fit_transform(df_testing)

###Diviser le testing set en x_test et y_test
x_test = []
y_test = []
for i in range(10, df_testing_array.shape[0]):
  x_test.append(df_testing_array[i-10 : i ])
  y_test.append(df_testing_array[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

###Prédictions pour x_test
y_pred = model.predict(x_test)
#Annuler le scaling
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_testing = scaler.inverse_transform(y_test.reshape(-1, 1))


###Visionner la courbe des prédiction du cours de clôture VS le cours du prix de clôture réels
st.subheader('Valeurs prédites VS valeurs réelles')
fig3=plt.figure(figsize=(15,6))
plt.plot(y_testing, 'b', label = 'Prix réel')
plt.plot(y_pred, 'r', label='Prix prédit')
plt.xlabel('Temps')
plt.ylabel('Prix de clôture (EUR)')
plt.legend()
st.pyplot(fig3)

###Prédiction du prix de clôture pour demain
x = []
x.append(y_test[-10:])
x=np.array(x)
x_reshaped = np.reshape(x, (1, 10, 1))
pred_tomorrow = model.predict(x_reshaped)
pred_tomorrow_n = scaler.inverse_transform(pred_tomorrow.reshape(-1, 1))
st.subheader('Prédiction du prix de cloture pour demain')
next_day = dt.datetime.now() + dt.timedelta(days=1)
tomorrow= next_day.strftime("%Y-%m-%d")
st.write("Le prix de cloture prédit (en euro) pour le {} est : {:.2f}".format(tomorrow,float(pred_tomorrow_n)))
