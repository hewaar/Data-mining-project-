# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:50:59 2023

@author: hewaa
"""
#Stap 1
# Dataframes maken en missende en dubbele waardes schoonmaken 

import os
import pandas as pd

csv_folder_path = "C:\\Users\\hewaa\\Documents\\Minor 3DiB\\Data Mining F1 Data"

# Lijst om DataFrames op te slaan
dataframes = {}

# Lijst van CSV-bestanden in de map
csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

# Loop door elk CSV-bestand en laad het in een DataFrame
for csv_file in csv_files:
    try:
        # Genereer een sleutelnaam op basis van het bestandsnaam zonder extensie
        key = os.path.splitext(csv_file)[0]
        
        # Lees het CSV-bestand in een DataFrame
        df = pd.read_csv(os.path.join(csv_folder_path, csv_file))
        
        # Verwijder dubbele waarden
        df = df.drop_duplicates()
        
        # Behandel ontbrekende waarden door ze te verwijderen (je kunt ook andere methoden gebruiken om ze in te vullen)
        df = df.dropna()
        
        # Sla het DataFrame op in de lijst met DataFrames
        dataframes[key] = df
        
        print(f"{key} ingelezen. Rijen na reiniging: {len(df)}, Kolommen: {len(df.columns)}")
    except Exception as e:
        print(f"Fout bij het inlezen van {csv_file}: {str(e)}")
        

# 
import pandas as pd

# Maak een voorbeeld DataFrame met een categorisch kenmerk 'color'
data = {'color': ['rood', 'blauw', 'groen', 'rood', 'blauw']}
df = pd.DataFrame(data)

# Voer one-hot encoding uit
df_encoded = pd.get_dummies(df, columns=['color'], prefix=['color'])

# Het resulterende DataFrame ziet er zo uit:
#    color_blauw  color_groen  color_rood
# 0            0           0          1
# 1            1           0          0
# 2            0           1          0
# 3            0           0          1
# 4            1           0          0

from sklearn.preprocessing import LabelEncoder

# Maak een voorbeeld DataFrame met een categorisch kenmerk 'size'
data = {'size': ['klein', 'groot', 'medium', 'klein', 'groot']}
df = pd.DataFrame(data)

# Maak een LabelEncoder
label_encoder = LabelEncoder()

# Voer label encoding uit
df['size_encoded'] = label_encoder.fit_transform(df['size'])

# Het resulterende DataFrame ziet er zo uit:
#      size  size_encoded
# 0    klein            0
# 1    groot            2
# 2   medium            1
# 3    klein            0
# 4    groot            2

# Ggevens scheiden in traning en test data

from sklearn.model_selection import train_test_split

# De sleutel wordt hier direct gedefinieerd
jouw_dataframesleutel = os.path.splitext(csv_file)[0]  # Dit is je DataFrame sleutel

# Definieer je functies (features) en doelvariabele (target) op basis van je gecombineerde dataset
features = dataframes[jouw_dataframesleutel].drop('resultId', axis=1)  # Laat 'resultId' weg uit features
target = dataframes[jouw_dataframesleutel]['resultId']  # Gebruik 'resultId' als de doelvariabele

# Splits de gegevens in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Nu heb je X_train, X_test, y_train en y_test om mee te werken

#random forest regressiemodel 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Vervang dit pad door het pad naar je 'results' CSV-bestand
csv_file_path = "C:\\Users\\hewaa\\Documents\\Minor 3DiB\\Data Mining F1 Data\\results.csv"

# Lees de 'results' dataset in
data = pd.read_csv(csv_file_path)

# Definieer je functies (features) en doelvariabele (target)
X = data.drop('resultId', axis=1)  # Laat 'resultId' weg uit features
y = data['resultId']  # Gebruik 'resultId' als de doelvariabele

# Schoonmaak van de dataset (optioneel)
X.replace("\\N", np.nan, inplace=True)  # Vervang "\\N" door NaN voor ontbrekende waarden
X = X.dropna()  # Verwijder rijen met ontbrekende waarden

# Split de gegevens in trainings- en testsets (90% trainingsdata, 10% testdata)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Maak een instantie van het regressiemodel
regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # Pas het aantal bomen en andere hyperparameters aan indien nodig

# Train het model op de trainingsgegevens
regressor.fit(X_train, y_train)

# Voorspel de resultaten voor de testgegevens
y_pred = regressor.predict(X_test)

# Bereken de RMSE (Root Mean Square Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Square Error (RMSE): {rmse}")












