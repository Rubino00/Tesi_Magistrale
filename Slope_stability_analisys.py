
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt


# pd.set_option ("display.max_columns" , None)
df = pd.read_csv ("C:\\Users\\fraru\\OneDrive\\Desktop\\programmi\\Clone_python\\Tesi_magistrale\\Datastes\\slope_stability_dataset.csv")

def info_df (df:pd.DataFrame):
    print (df.info ())
    print (df.isnull ().sum ())
    print (df.head (30))

sbn.scatterplot (data = df , x = "Slope Angle (°)" , y = "Factor of Safety (FS)")
plt.show ()