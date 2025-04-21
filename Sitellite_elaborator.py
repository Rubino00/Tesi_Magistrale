
import pandas as pd
import numpy as np
import seaborn as sbn
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
file_path = "C:\\Users\\fraru\OneDrive\Desktop\programmi\Clone_python\Tesi_magistrale\Datastes\Sentinel_images\sentinel image.tif"
raster = rasterio.open (file_path)

# definiamo una funzione che ha lo scopo di fornire informazioni generali al raster
def info_raster (raster):
    print (f"numero bande: {raster.count}")
    print (f"dimensiona immagine: {raster.width} x {raster.height}")
    print (f"sistema di riferimento: {raster.crs}")
    print (f"formato del dato: {raster.dtypes}")
    print (f"attributo transform: {raster.transform}")

def raster_to_data (raster_array:np.array , band_list:list):
    reshaped_array = raster_array.reshape (raster_array.shape [0] , -1).T
    df_raster = pd.DataFrame (reshaped_array , columns = band_list)
    return reshaped_array


band_list = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8" , "SCL"]
raster_array = raster.read ()
reshaped_array = raster_to_data (raster_array , band_list)



