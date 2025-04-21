
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.linear_model import LinearRegression
import statistics as st
import scipy as sc

path_file = "C:\\Users\\fraru\OneDrive\Desktop\programmi\Clone_python\Tesi_magistrale\Datastes\Diabetes_dataset\diabetes_dataset.csv"
df = pd.read_csv (path_file)

def info_df (df:pd.DataFrame):
    print (df.info ())
    print (df.isnull ().sum ())
    print (df.head (30))

def info_columns (df:pd.DataFrame):
    for column in df.columns:
        print (f"colonna: {column}")
        print (df [column].describe ())
        print ()

etnia_list = list (df ["Ethnicity"].unique ())
mean_bmi = []

for etnia in etnia_list:
    mean_bmi.append (df [df ["Ethnicity"] == etnia] ["BMI"].mean ())

q_1 = df ["BMI"].quantile (0.25)
q_2 = df ["BMI"].quantile (0.5)
q_3 = df ["BMI"].quantile (0.75)
iqr = q_3 - q_1

min_val = q_1 - 1.5 * iqr
max_val = q_3 + 1.5 * iqr

print (f"primo quartile: {q_1}")
print (f"mediana: {q_2}")
print (f"terzo quartile: {q_3}")
print (f"minimo valore teorico: {min_val}")
print (f"massimo valore teorico: {max_val}")
print (f"minimo valore reale: {df ['BMI'].min ()}")
print (f"massimo valore reale: {df ['BMI'].max ()}")


def calculate_zscore (df:pd.DataFrame , column:str):

    mean = df [column].mean ()
    std = df [column].std ()

    def z_score (row):
        zscore = abs ((row [column] - mean) / std)
        return zscore
    df [f"{column}_zscore"] = df.apply (z_score , axis = 1)

calculate_zscore (df , "BMI")
calculate_zscore (df , 'Waist_Circumference')
print (df [["BMI" , "BMI_zscore" , 'Waist_Circumference' , 'Waist_Circumference_zscore']])
plot = sbn.histplot (data = df , x = "BMI_zscore" , label = "BMI_zscore" , bins = 10)
plot = sbn.histplot (data = df , x = 'Waist_Circumference_zscore' , label = 'Waist_Circumference_zscore' , bins = 10)
plot.set_xlabel ("zscore")
plot.set_ylabel ("count")
plot.set_title ("distribuzione della variabile standardizzata")
plt.legend ()
plt.show ()






