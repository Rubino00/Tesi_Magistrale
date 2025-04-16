
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_path = "C:\\Users\\fraru\OneDrive\Desktop\programmi\Clone_python\Tesi_magistrale\Datastes\House_prices\House_prices_dataset.csv"
df = pd.read_csv (input_path)

# definiamo una funzione utile alla descrizione generale del dataframe
def info_df (df:pd.DataFrame):
    print (df.info ())
    print (df.isnull ().sum ())
    print (df.head (30))

# definiamo una funzione utile a una descrizione dettagliata di ogni colonna del dataframe
def info_columns (df:pd.DataFrame):
    for column in df.columns:
        print (f"colonna {column}:")
        print (df [column].describe ())
        print ()


# definiamo un nuovo dataframe contenente tutte le variabili ordinate in diverse colonne ogniuna relativa
# a diverse features. Le colonne in questione vengono elencate di seguito:
string_columns = " - CRIM per capita crime rate by town - ZN proportion of residential land zoned for lots over 25,000 sq.ft. - INDUS proportion of non-retail business acres per town - CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) - NOX nitric oxides concentration (parts per 10 million) - RM average number of rooms per dwelling - AGE proportion of owner-occupied units built prior to 1940 - DIS weighted distances to five Boston employment centres - RAD index of accessibility to radial highways - TAX full-value property-tax rate per $10,000 - PTRATIO pupil-teacher ratio by town - B 1000(Bk-0.63)^2 where Bk is the proportion of blacks by town - LSTAT % lower status of the population - MEDV Median value of owner-occupied homes in $1000's"
list_string_columns = string_columns.split (" - ")

# Scriviamo una lista contenente tutti i nomi delle colonne del nuovo database. Il nome delle colonne viene estratto
# direttamente dalla strinfa "string_columns"
column_list = []
for complete_column in list_string_columns:
    column = complete_column.split (" ") [0]
    column_list.append (column)
column_list = column_list [1:]

# ogni record del nostro dataframe grezzo è una stringa di questo tipo:
# '0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00'
# la stringa deve essere divisa e i numero devono essere estratti. Successivamente i vari numeri devono essre
# organizzati all'interno di un dizionario che conterrà come chiave il nome della colonna contenuta all'interno
# della lista "column_list". La chiave sarà associata a una lista di valori riferiti ad ogni record e a quella colonna

# costruiamo prima il dizionario
new_df_dict = {}
for column in column_list:
    new_df_dict [column] = []

# definiamo una funzione che ci permette di manipolare ogni rigo del nostro dataframe
def separate_string (string_row:str):

    # definisco una lista contenente tutti i numeri ordinati e convertiti in float
    temp_list_string = string_row.split (" ")
    list_num = []
    for elem in temp_list_string:
        if elem == "":
            pass
        else:
            list_num.append (float (elem))

    # definisco una funzione che aggiunga al dizionario tutti i nuovi valori di quel record
    i = 0
    while i < len (column_list):
        new_df_dict  [column_list [i]].append (list_num [i])
        i = i + 1
    return list_num

# applichiamo ad ogni riga del nostro dataframe la funzione "separate_string" per scrivere il nostro dizionario
df.iloc [: , 0].apply (separate_string)
new_df = pd.DataFrame (new_df_dict)
info_df (new_df)
info_columns (new_df)

