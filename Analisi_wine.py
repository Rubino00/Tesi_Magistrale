
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# stampiamo alcune informazioni del nostro dataframe
def info_df (df:pd.DataFrame):
    print (df.info ())
    print (df.isnull ().count ())
    print (df.head ())

# analiziamo la validazione dei dati
def validation (results , y_test):
    accuracy = accuracy_score (results , y_test)
    accuracy = round (accuracy * 100 , 2)
    conf_matr = confusion_matrix (results , y_test)
    class_rep = classification_report (results , y_test)

    print (accuracy)
    print (class_rep)
    sbn.heatmap (conf_matr)
    plt.show ()

# carichiamo il dataset wine da sci-kit learn
wine = load_wine ()
df = pd.DataFrame (data = wine.data , columns = wine.feature_names)
df ["target"] = wine.target
info_df (df)
x_set = df [df.columns [:-1]]
y_set = df ["target"]

print (x_set)
print (y_set)

# creiamo i dati di apprendimento e di validazione
x_train , x_test , y_train , y_test = train_test_split (x_set , y_set , test_size = 0.50 , random_state = 42)

# scaliamo i valori contenuti nelle colonne di "x_train" e "x_test"
scaler = StandardScaler ()
x_train = scaler.fit_transform (x_train)
x_test = scaler.transform (x_test)

# costruiamo il nostro classificatore e classifichiamo i vari tipi di vino
knn = KNeighborsClassifier (n_neighbors = 5)
knn.fit (x_train , y_train)
results = knn.predict (x_test)

# verifichiamo le performance del nostro modello
validation (results , y_test)


