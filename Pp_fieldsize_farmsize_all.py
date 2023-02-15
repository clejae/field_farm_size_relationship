#Script for preparing variables at field level for Brandenburg
#variables: slope, elevation, ###Crop Sequence Type###, useable field capacity, soil water, 
#spatial belonging to hexagon in hexagongrid, former DDR-member or not


#Franziska Frankenfeld
#----------------------------------------------------------------------------------------------------
##import required modules
import pandas as pd;
import geopandas as gp;

from osgeo import gdal,osr;
import numpy as np;
import rasterio;
from rasterstats import zonal_stats;
from rasterio.mask import mask
from scipy import stats
import shapely
from shapely import wkt
#----------------------------------------------------------------------------------------------------

#data input directory
data_in = "//141.20.140.92/SAN_Projects/FORLand/Field_farm_size_relationship/data/"
#data output directory
data_out = "//141.20.140.92/SAN_Projects/FORLand/Field_farm_size_relationship/data/"

#----------------------------------------------------------------------------------------------------

###VECTOR DATA PROCESSING

##read vector data
#invekos
inv_3035 = gp.read_file(data_in+"vector/IACS/IACS_ALL_2018_3035_validated.shp")
inv_3035 = inv_3035.dropna()

#hexagon grid
hexa_3035 = gp.read_file(data_in+"vector/grid/hexagon_grid_GER_5km_3035.shp")

#data info
print(hexa_3035[0:10])
print('-------------')
print(inv_3035[0:10])

#data info
print(type(inv_3035)) 
print(type(hexa_3035))
print(inv_3035.crs) 
print(hexa_3035.crs) 

#----------------------------------------------------------------------------------------------------

###assign if former East or West Germany member

##get info on existing laender-codes in data by examining field_id
#put field_id column into pandas series format
series = inv_3035['field_id'].squeeze()
#use split function of pandas series to split string of field_id right after the laender-code letters in the beginning
new = series.str.split(pat="_", n=1, expand=True)
#print unique laender-codes in data
print(new[0].unique())

#add new column with value based on laender code, BB Brandeburg, TH Thueringen, LS Lower Saxony, BV Bavaria, SA Sachsen Anhalt, 
#create function
def f(row):
    if row['field_id'].startswith("BB") or row['field_id'].startswith("TH") or row['field_id'].startswith("SA"):
        val = 1
    elif row['field_id'].startswith("LS") or row['field_id'].startswith("BV"):
        val = 0
    else:
        val = 999
    return val

#apply function
inv_3035["FormerDDRmember"] = inv_3035.apply(f, axis=1)
print(inv_3035)
print(inv_3035.FormerDDRmember.unique())
len(inv_3035[inv_3035['FormerDDRmember']==1])
len(inv_3035[inv_3035['FormerDDRmember']==0])
len(inv_3035[inv_3035['FormerDDRmember']==999]) 

#----------------------------------------------------------------------------------------------------
#calculate centroids of fields

#do copy of invekos data to avoid problems with changing of datatypes and incompatibility
inv_c = inv_3035.copy()
#calculate field area sizes
#inv_c.loc["fieldsize"] = inv_c.area

#approach by Clemens to get centroids
inv_c["centroids"] = inv_c["geometry"].centroid
centroids = inv_c[["field_id", "centroids"]].copy()
centroids.rename(columns={"centroids": "geometry"}, inplace=True)
#drop centroids in invekos dataframe
inv_c.drop(["centroids"],axis=1,inplace=True)

## Create a centroids layer and add the hex id to the centroids
centroids = gp.GeoDataFrame(centroids, crs=3035)
centroids = gp.sjoin(centroids, hexa_3035, how="left", op="intersects")

#rename id to hex_id for clarity
centroids.rename(columns={'id': 'hexa_id'}, inplace=True)

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

###RASTER DATA PROCESSING

#calculate average slope per field

with rasterio.open(data_in+"raster/Slope_3035.tif") as src:
    affine = src.transform #using affine transformation matrix
    array = src.read(1)
    print(src.nodatavals) #print no data value
    print(array)
    ndval = src.nodatavals[0]
    #array = array.astype('float64')
    array[array==ndval] = np.nan #set no data values to appropriate na-value format
    print(array)
    print(np.isnan(array).sum()) #127 487 852
    print(array.size) #912 145 488 13% of all Pixels are na
    
    df_zonal_stats = pd.DataFrame(zonal_stats(inv_3035, array, all_touched = True, nodata=np.nan, affine=affine, stats=["mean"]))
    print(df_zonal_stats)
    print(df_zonal_stats['mean'].isnull().sum()) #8112
    print(len(df_zonal_stats)) #1624222
    
    #na_val = df_zonal_stats[df_zonal_stats['mean'].isnull()]
    #print(na_val)
    #gdf_na = pd.concat([inv_3035, na_val], axis=1)
    #gdf_na.to_file(data_in+"test_run2/slopeNA_only", driver="ESRI Shapefile")

#adding statistics back to original GeoDataFrame
slope = pd.concat([inv_3035, df_zonal_stats], axis=1)

#rename columns
slope.rename(columns={'mean': 'SlopeAvrg'}, inplace=True)
print(slope)

#write shapefile
slope.to_file(data_in+"test_run2/slopeAvrg", driver="ESRI Shapefile")

#----------------------------------------------------------------------------------------------------

#calculate average elevation per field

with rasterio.open(data_in+"raster/Dem_3035.tif") as src:
    affine = src.transform #using affine transformation matrix
    array = src.read(1)
    print(src.nodatavals) #print no data value
    print(array)
    ndval = src.nodatavals[0]
    #array = array.astype('float64') 
    array[array==ndval] = np.nan #set no data values to appropriate na-value format
    print(array)  
    
    df_zonal_stats = pd.DataFrame(zonal_stats(inv_3035, array, all_touched = True, nodata=np.nan, affine=affine, stats=["mean", "std"]))
    print(df_zonal_stats)

#adding statistics back to original GeoDataFrame
dem = pd.concat([inv_3035, df_zonal_stats], axis=1)

#rename columns
dem.rename(columns={'mean': 'ElevationA', 'std': 'sdElevatio'}, inplace=True)
print(dem)

#write shapefile
dem.to_file(data_in+"test_run2/elevationAvrg", driver="ESRI Shapefile")

#----------------------------------------------------------------------------------------------------

#determine most common crop sequence type per field
with rasterio.open(data_in+"raster/CST/all_2012-2018_CropSeqType_3035.tif") as src:
    affine = src.transform #using affine transformation matrix
    array = src.read(1)
    print(array)
    print(src.nodatavals) #print no data value
    
    #ndval = src.nodatavals[0]
    #array = array.astype('float')
    #array[array==ndval] = np.nan #set no data values to appropriate na-value format
    #print(array)
    df_zonal_stats = pd.DataFrame(zonal_stats(inv_3035, array,nodata=255.0, all_touched = True, affine=affine, stats=["majority"]))
    print(df_zonal_stats)

# adding statistics back to original GeoDataFrame
cst = pd.concat([inv_3035, df_zonal_stats], axis=1)

#rename columns
cst.rename(columns={'majority': 'CstAvrg'}, inplace=True)
print(cst)

#write shapefile
cst.to_file(data_in+"test_run2/CstAvrg", driver="ESRI Shapefile")

# ----------------------------------------------------------------------------------------------------

#calculate average field usage capacity (nutzbare Feldkapazitaet NFKW)

with rasterio.open(data_in+"raster/NFKWe1000_250_3035.tif") as src:
    affine = src.transform #using affine transformation matrix
    array = src.read(1)
    print(src.width)
    print(src.height) #Total 8 665 984 pixels same as in Qgis
    
    print(src.nodatavals) #print no data value -3.4028230607370965e+38 #passt das mit float64?
    #print(array)
    
    print(np.isnan(array).sum())#0, weil noch no data value -3.4028...
    ndval = src.nodatavals[0]
    array = array.astype('float64') 
    array[array==ndval] = np.nan #set no data values to appropriate na-value format
    print(np.isnan(array).sum()) #4 046 041 same as in Qgis
    print(array.size) #8 665 984, 46% of all pixels are na
    print(array)  
    
    df_zonal_stats = pd.DataFrame(zonal_stats(inv_3035, array, affine=affine, nodata=np.nan, all_touched = True, stats=["mean"]))
    print(df_zonal_stats)
    print(df_zonal_stats['mean'].isnull().sum()) #1 020  840
    print(len(df_zonal_stats)) #1 624 222

#adding statistics back to original GeoDataFrame
nfk = pd.concat([inv_3035, df_zonal_stats], axis=1)

#rename columns
nfk.rename(columns={'mean': 'NfkAvrg'}, inplace=True)
print(nfk)

#write shapefile
nfk.to_file(data_in+"test_run2/NfkAvrg", driver="ESRI Shapefile")

#----------------------------------------------------------------------------------------------------

#calculate average soil water (Austausch Bodenwasser AHAACGL)

with rasterio.open(data_in+"raster/AHACGL1000_250_3035.tif") as src:
    affine = src.transform #using affine transformation matrix
    array = src.read(1)
    print(src.nodatavals) #print no data value
    print(array)
    ndval = src.nodatavals[0]
    #array = array.astype('float64') 
    array[array==ndval] = np.nan #set no data values to appropriate na-value format
    print(array)  
    
    df_zonal_stats = pd.DataFrame(zonal_stats(inv_3035, array, affine=affine, all_touched = True, nodata=np.nan, stats=["mean"]))
    print(df_zonal_stats)

#adding statistics back to original GeoDataFrame
aha = pd.concat([inv_3035, df_zonal_stats], axis=1)

#rename columns
aha.rename(columns={'mean': 'AhaacglAvr'}, inplace=True)
print(aha)

#write shapefile
aha.to_file(data_in+"test_run2/AhaacglAvrg", driver="ESRI Shapefile")

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

#in preparation for the final merge, we first have to get rid of the NAs in field_id, which were produced during the last steps when calculating the raster statistics per field (raster values where were no field geometries were also put in the df and therefore there are a lot of NA values in field_id)


#print length of dataframes
#print(len(slope))

#check for nas
#print(slope[slope.duplicated(subset=['field_id'],keep=False)]) #some observations have NA in field id and only info from raster. We can drop them because they are not part of invekos

#drop nas
slope_new = slope.dropna(subset = ['field_id'])
print(len(slope_new))

dem_new = dem.dropna(subset = ['field_id'])
print(len(dem_new))

cst_new = cst.dropna(subset = ['field_id'])
print(len(cst_new))

nfk_new = nfk.dropna(subset = ['field_id'])
print(len(nfk_new))

aha_new = aha.dropna(subset = ['field_id'])
print(len(aha_new))

#merge
merge1 = pd.merge(slope_new, cst_new[["field_id", "CstAvrg"]], how="left", on="field_id", validate ="one_to_one")
print(len(merge1))

merge2 = pd.merge(nfk_new, aha_new[["field_id", "AhaacglAvr"]], how="left", on="field_id", validate ="one_to_one")

merge22 = pd.merge(merge2, dem_new[["field_id", "ElevationA", "sdElevatio"]], how="left", on="field_id", validate ="one_to_one")
print(len(merge22))

#merge final
final_merge = pd.merge(merge1, merge22[["field_id", "NfkAvrg", "AhaacglAvr","ElevationA", "sdElevatio"]], how="left", on="field_id", validate ="one_to_one")
print(len(final_merge))
print(final_merge)

#move geometry column to the end of df
column_to_move = final_merge.pop("geometry")
# insert column with insert(location, column_name, column_value)
final_merge.insert(11, "geometry", column_to_move)
print(final_merge)

#write shapefile
final_merge.to_file(data_in+"test_run2/all_predictors", driver="ESRI Shapefile")


#----------------------------------------------------------------------------------------------------
#add centroids info

print(len(final_merge))
print(len(centroids))
                          
## Add hexagon id to iacs data by joining the centroids values
inv_compl = pd.merge(final_merge, centroids[["field_id", "hexa_id"]], how="left", on="field_id")
print(inv_compl)

#move geometry column to the end of df
column_to_move = inv_compl.pop("geometry")
# insert column with insert(location, column_name, column_value)
inv_compl.insert(12, "geometry", column_to_move)
print(inv_compl)

#----------------------------------------------------------------------------------------------------

#do 10% subset of whole dataframe
inv_10perc = inv_compl.sample(frac= 0.1)

#write this subset to csv
inv_10perc.to_csv(data_out+"test_run2/10perc_final_inv.csv", index=False)

#write complete dataset to shapefile
inv_compl.to_file(data_out+"test_run2/final_inv_compl", driver="ESRI Shapefile")


