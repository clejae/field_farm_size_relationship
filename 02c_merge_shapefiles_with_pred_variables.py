#Script to merge the predictor shapefiles and get one big dataframe with all the data

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

#data input directory
data_in = "//141.20.140.92/SAN_Projects/FORLand/Field_farm_size_relationship/data/"
#data output directory
data_out = "//141.20.140.92/SAN_Projects/FORLand/Field_farm_size_relationship/data/"

#open shapefiles
slope = gp.read_file(data_in+"test_run2/slopeAvrg.shp")
dem = gp.read_file(data_in+"test_run2/elevationAvrg.shp")
cst = gp.read_file(data_in+"test_run2/CstAvrg.shp")
nfk = gp.read_file(data_in+"test_run2/NfkAvrg.shp")
aha = gp.read_file(data_in+"test_run2/AhaacglAvrg.shp")

#print them
print(slope)
print(dem)
print(cst)
print(aha)
print(nfk)


#show datatypes of columns
display(slope.dtypes)
display(dem.dtypes)


"""
###only testing things

#try first merge
alle = pd.merge(slope, cst[["field_id", "CstAvrg"]], how="left", on="field_id")

#length in first merge is already longer than the single dataframes. Why? one to many instead of one to one match
len(all)
len(dem)
len(slope)
len(cst)

#all = pd.merge(all, nfk[["field_id", "NfkAvrg"]], how="left", on="field_id")
#all = pd.merge(all, aha[["field_id", "AhaacglAvr"]], how="left", on="field_id")
#all = pd.merge(all, dem[["field_id", "ElevationA"]], how="left", on="field_id")


#try with inner join
#how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False,
alle = pd.merge(slope, cst[["field_id", "CstAvrg"]], how="inner", on="field_id")
len(alle) #still more rows

#try with join
#all = slope.join( cst, on="field_id", how='left') #doesn't work, program suggests to use concat
#len(all)

#try with concat
#alle = pd.concat([slope, cst], axis=1) #when doing this, pc gives up and I have to reload EVERYTHiNG
#len(alle)


#related to post, try merge with one-to-one
sub_slope = slope[0:500]
len(sub_slope)
sub_cst = cst[0:500]
alle = pd.merge(sub_slope, sub_cst[["field_id", "CstAvrg"]], how="left", on="field_id", validate ="one_to_one")
len(alle)

alle = pd.merge(slope, cst[["field_id", "CstAvrg"]], how="left", on="field_id", validate ="one_to_one")
len(alle) #merge keys are not unique in either left or right dataset, not a one-to-one merge
print(slope[slope.duplicated(subset=['field_id'],keep=False)]) #some have field_id = none but value in slope, also don't have geometry
print(cst[cst.duplicated(subset=['field_id'],keep=False)])

"""
#print length of dataframes
print(len(cst))
print(len(slope))

#check for nas
print(cst[cst.duplicated(subset=['field_id'],keep=False)]) #some observations have NA in field id and only info from raster. We can drop them because they are not part of invekos

#drop nas
slope_new = slope.dropna(subset = ['field_id'])
print(len(slope_new))

cst_new = cst.dropna(subset = ['field_id'])
print(len(cst_new))

dem_new = dem.dropna(subset = ['field_id'])
print(len(dem_new))

nfk_new = nfk.dropna(subset = ['field_id'])
print(len(nfk_new))

aha_new = aha.dropna(subset = ['field_id'])
print(len(aha_new))

#merge
merge1 = pd.merge(slope_new, cst_new[["field_id", "CstAvrg"]], how="left", on="field_id", validate ="one_to_one")
print(len(merge1))

merge2 = pd.merge(nfk_new, aha_new[["field_id", "AhaacglAvr"]], how="left", on="field_id", validate ="one_to_one")

merge22 = pd.merge(merge2, dem_new[["field_id", "ElevationA", "sdElevatio"]], how="left", on="field_id", validate ="one_to_one")
print(len(alle3))

#merge alle, alle3
final_merge = pd.merge(merge1, alle3[["field_id", "NfkAvrg", "AhaacglAvr","ElevationA", "sdElevatio"]], how="left", on="field_id", validate ="one_to_one")
print(len(final_merge))
print(final_merge)

#move geometry column to the end of df
column_to_move = final_merge.pop("geometry")

# insert column with insert(location, column_name, column_value)
final_merge.insert(11, "geometry", column_to_move)
print(final_merge)

#write shapefile
final_merge.to_file(data_in+"test_run2/all_predictors", driver="ESRI Shapefile")