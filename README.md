# DeepLearningProject_CryptoForetell
This is a web application for predicting cryptocurrency closing prices

# CryptoForetell : Predicting the value of cryptocurrencies
La version française est ci-dessous (après la version anglaise)

## Description
CryptoForetell is a web application that allows users to visualize historical data and predict the closing prices of cryptocurrencies. 
The application uses data provided by the CryptoCompare API and implements an LSTM model to make predictions.

## Main Features

- Display of historical closing prices of cryptocurrencies.
- Calculation and visualization of 10 and 30-day moving averages.
- Predictions of future closing prices of cryptocurrencies for the next days.
- Intuitive user interface with Streamlit for a smooth user experience.

## Project Structure

The project is organized as follows:

- **app.py**: The main file of the Streamlit application that contains the code to load data, make predictions, and display the user interface.
- **CryptoForetell.ipynb**: The Jupyter Notebook file containing the development steps, exploratory data analysis, and LSTM model construction.
- **cryptoforetellmodel.h5**: The pre-trained LSTM model used to predict cryptocurrency closing prices.
- **SelectCrypto.py**: This program downloads cryptocurrency data from the CryptoCompare API and stores it in the "crypto_info.csv" file.
- **FilteredCrypto.py**: This is a Python code that filters cryptocurrency data from the "crypto_info.csv" file to keep usable records.
- **crypto_info_filtered.csv**: The CSV file containing filtered information about cryptocurrencies used in the application.
- **requirements.txt**: The file containing all Python dependencies needed to run the application.

## Installation and Running Locally

1. Clone this GitHub repository to your local machine:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git
2. Install the required dependencies by running the following command: pip install -r requirements.txt
3. Run the application using the following command in the project directory: streamlit run app.py
4. The application will launch in your default web browser.

## How to Use the Application
1. **Select Cryptocurrency**: Choose the cryptocurrency you want to visualize data and predictions for.
2. **Select Dates**: Choose the time period for which you want to visualize data and make predictions.
3. **Data Visualization**: Explore interactive charts to view historical closing prices, moving averages, and predictions.
4. **Predictions for Future Days**: Get predictions of closing prices for the upcoming days and track market trends.

## Authors
This project was developed by NGARI LENDOYE Alix.

# CryptoForetell : Prédiction de la valeur des cryptomonnaies
English version is above (before french version)

## Description
CryptoForetell est une application web qui permet aux utilisateurs de visualiser les données historiques et de prédire les prix de clôture des cryptomonnaies. 
L'application utilise les données fournies par l'API CryptoCompare et met en œuvre un modèle LSTM pour effectuer les prédictions.

## Fonctionnalités principales

- Affichage des prix de clôture historiques des cryptomonnaies.
- Calcul et visualisation des moyennes mobiles sur 10 et 30 jours.
- Prédictions des prix de clôture futurs des cryptomonnaies pour les prochains jours.
- Interface utilisateur intuitive avec Streamlit pour une expérience utilisateur fluide.

## Structure du projet

Le projet est organisé de la manière suivante :

- **app.py**: Le fichier principal de l'application Streamlit qui contient le code pour charger les données, effectuer les prédictions et afficher l'interface utilisateur.
- **CryptoForetell.ipynb**: Le fichier Jupyter Notebook contenant les étapes de développement, l'analyse exploratoire des données et la construction du modèle LSTM.
- **cryptoforetellmodel.h5**: Le modèle LSTM pré-entraîné utilisé pour prédire les prix de clôture des cryptomonnaies.
- **SelectCrypto.py**: Ce programme télécharge les données sur les cryptomonnaies depuis l'API CryptoCompare et les stocke dans le fichier CSV "crypto_info.csv".
- **FiltredCrypto.py**: Il s'agit d'un code Python qui filtre les données sur les cryptomonnaies présentes dans le fichier "crypto_info.csv" afin de conserver les enregistrements exploitables.
- **crypto_info_filtered.csv**: Le fichier CSV contenant les informations filtrées sur les cryptomonnaies utilisées dans l'application.
- **requirements.txt**: Le fichier contenant toutes les dépendances Python nécessaires pour exécuter l'application.

## Installation et exécution en local

1. Clonez ce dépôt GitHub sur votre machine locale :
   ```bash
   git clone https://github.com/VOTRE_UTILISATEUR_GITHUB/NOM_DU_REPO.git
2. Installez les dépendances requises en exécutant la commande suivante : "pip install -r requirements.txt"
3. Exécutez l'application en utilisant la commande suivante dans le dossier du projet: "streamlit run app.py"
4. L'application se lancera dans votre navigateur par défaut.

## Comment utiliser l'application

1. **Sélection de la cryptomonnaie**: Choisissez la cryptomonnaie dont vous souhaitez visualiser les données et les prédictions.
2. **Sélection des dates**: Sélectionnez la période de temps pour laquelle vous souhaitez visualiser les données et effectuer les prédictions.
3. **Visualisation des données**: Explorez les graphiques interactifs pour voir les prix de clôture historiques, les moyennes mobiles et les prédictions.
4. **Prédictions pour les prochains jours**: Obtenez les prédictions des prix de clôture pour les jours à venir et suivez les tendances du marché.

## Auteurs
Ce projet a été développé par NGARI LENDOYE Alix 
