

import pandas as pd
import numpy as np
import seaborn as sbn
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

file_path_raster = "C:\\Users\\fraru\OneDrive\Desktop\programmi\Clone_python\Tesi_magistrale\Datastes\Sentinel_images\sentinel image.tif"
raster = rasterio.open (file_path_raster)

file_path_vectorial = "C:\\Users\\fraru\OneDrive\Desktop\Qgis\cartine shapefile e QGIS\cartina dell'Italia x comuni\Com2011_WGS84\Com2011_WGS84.shp"
data_vector = gpd.read_file (file_path_vectorial)

# definiamo una funzione che ha lo scopo di visualizzare il generico dizionario
def print_dict (dict_gen: dict):
    for key in dict_gen:
        print (f"{key}: {dict_gen [key]}")


# definiamo una funzione in grado di stampare informazioni di un dataframe
def info_df (df):
    print (df.info ())
    print (df.isnull ().sum ())
    print (df.head (30))

# definiamo una funzione per convertire il raster in dataframe di pandas riportando le bande alle colonne
def raster_to_data (raster_array: np.array , band_list: list):
    reshaped_array = raster_array.reshape (raster_array.shape [0] , -1).T
    df_raster = pd.DataFrame (reshaped_array , columns = band_list)
    return df_raster


# definiamo una funzione che trasformi il dataframe in un array
def data_to_raster (df: pd.DataFrame , row , column):
    count = len (df.columns)
    return df.to_numpy ().T.reshape ((count , row , column))


# definiamo una funzione di normalizzazione, i valori delle bande si rappresenteranno con un valore
# compreso tra 0 e 1
def normalize_band (band):
    max_val = band.max ()
    normalized_band = band / max_val
    return normalized_band

# elenco colonne: 'OBJECTID', 'COD_REG', 'COD_PRO', 'COD_ISTAT', # 'PRO_COM', 'NOME', 'SHAPE_Leng',
# 'SHAPE_Area', 'geometry'

