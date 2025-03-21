
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

wine = load_wine ()
df = pd.DataFrame (data = wine.data , columns = wine.feature_names)
df ["target"] = wine.target
class_list = list (df ["target"].unique ())
count_list = []

for classe in class_list:
    count_list.append (df [df ["target"] == classe] ["target"].count ())

sbn.barplot (x = class_list , y = count_list)
plt.show ()