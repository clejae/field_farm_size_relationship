#Script for preparing variables at field level for Brandenburg
#variables at field level: slope, elevation, ###Crop Sequence Type Majority###, useable field capacity, soil water,  #former DDR-member or not, spatial belonging to hexagon in hexagongrid
#variables at hexagon level: Number of fields in hexagon,sd ELevtation /ruggedness per hexagon, proportion of agricultural areas per hexagon

#Franziska Frankenfeld
#----------------------------------------------------------------------------------------------------
##import required modules
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
import warnings
from scipy.sparse import csr_matrix
from rasterio import features
from rasterio.io import MemoryFile
import math
from rtree import index
from joblib import Parallel, delayed

# from osgeo import gdal,osr
# import io
# import time
# import matplotlib.pyplot as plt
# from rasterio.mask import mask
# from scipy import stats
# import shapely
# from shapely import wkt

global iacs
global rtree_index

#----------------------------------------------------------------------------------------------------
WD = "//141.20.140.92/SAN_Projects/FORLand/Field_farm_size_relationship/data/"
os.chdir(WD)

#----------------------------------------------------------------------------------------------------

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def zonal_stats_alt(gpd, rst_fn, stats=["mean"]):
    rst = rasterio.open(rst_fn)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    # gpd.geometry = gpd.buffer(1000)
    # Create an in-memory file-like object
    memfile = MemoryFile()

    with memfile.open(driver='GTiff', width=meta['width'], height=meta['height'], count=1, dtype=meta['dtype'],
                      crs=meta['crs'], transform=meta['transform'], compress='lzw') as dataset:
        out_arr = dataset.read(1)

        # Create a generator of geom, value pairs to use in rasterizing
        # index + 1 because we fill with zeros -> zero is the no data value; float not possible because int needed for
        # sparse indices
        shapes = ((geom, value) for geom, value in zip(gpd.geometry, gpd.index+1))

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=dataset.transform, all_touched=True)
        dataset.write_band(1, burned)
        # Read the modified data from the in-memory dataset
        arr = dataset.read(1)

    sparse_idx = get_indices_sparse(arr.astype(int))
    with rasterio.open("raster/Slope_3035.tif") as src:
        array = src.read(1)
        ndval = src.nodatavals[0]
        print('NDVAL: ', ndval)
        array[array == ndval] = np.nan  # set no data values to appropriate na-value format

    mean_values = []
    std_values = []
    for id, indices_field in enumerate(sparse_idx):
        # skip zero because no data value
        if id == 0:
            continue
        else:
            std_values.append(np.nanstd(array[indices_field]))
            mean_values.append(np.nanmean(array[indices_field]))

    gpd['mean'] = mean_values
    gpd['std'] = std_values
    return gpd


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------
    ###VECTOR DATA PROCESSING
    ##read vector data
    # invekos
    iacs = gpd.read_file("vector/IACS/IACS_ALL_2018_with_grassland_recl_3035.shp")
    iacs.dropna(inplace=True)

    # hexagon grid
    hexa = gpd.read_file("vector/grid/hexagon_grid_GER_5km_3035.shp")

    # data info
    print("CRS IACS:", iacs.crs)
    print("CRS Hexagons:", hexa.crs)

    # ----------------------------------------------------------------------------------------------------
    # calculate field sizes and hexagon sizes

    # calculate field size
    iacs["fieldSizeM2"] = iacs.area
    # iacs.drop(["field_size"], axis=1, inplace=True)

    # calculate area size per hexagon
    hexa["hexaSizeM2"] = hexa.area

    # rename id to hex_id for clarity
    hexa.rename(columns={'id': 'hexa_id'}, inplace=True)

    # ----------------------------------------------------------------------------------------------------
    ###Count number of fields per farm
    df = iacs.groupby(['farm_id']).agg(
        fieldCount=pd.NamedAgg("field_id", "count")
    ).reset_index()

    # merge to invekos dataframe
    iacs = pd.merge(iacs, df, how="left", on="farm_id")

    # ----------------------------------------------------------------------------------------------------
    ###assign if former East or West Germany member

    iacs["federal_state"] = iacs["field_id"].apply(lambda x: x.split("_")[0])
    iacs['FormerDDRmember'] = 0
    iacs.loc[iacs["federal_state"].isin(["BB", "SA", "TH"]), "FormerDDRmember"] = 1

    print("Values for FormerDDRmember:", iacs.FormerDDRmember.unique())

    # ----------------------------------------------------------------------------------------------------

    # calculate centroids of fields
    iacs["centroids"] = iacs["geometry"].centroid
    centroids = iacs[["field_id", "centroids"]].copy()
    centroids.rename(columns={"centroids": "geometry"}, inplace=True)
    centroids = centroids.set_geometry("geometry")
    iacs.drop(columns=["centroids"], inplace=True)

    ## Create a centroids layer and add the hex id to the centroids
    centroids = gpd.GeoDataFrame(centroids, crs=3035)
    centroids = gpd.sjoin(centroids, hexa, how="left", op="intersects")

    # add centroids info to invekos dataframe
    # check if same length

    if len(iacs) == len(centroids):
        print("Merging IACS with centroids and hexagon information.")
        iacs = pd.merge(iacs, centroids[["field_id", "hexa_id"]], how="left", on="field_id")
    else:
        warnings.warn("IACS and centroids don't have the same length!", len(iacs), len(centroids))

    # ----------------------------------------------------------------------------------------------------
    def process_chunk(chunk, iacs):
        ## temp:
        # chunk = iacs.index[0:0 + 10]
        # index = chunk[1]
        ## creat spatial index
        spatial_index = iacs.copy().sindex
        intersected_list = []
        ## loop over rows
        for index in chunk:
            row = iacs.iloc[index]
            geom = row.geometry.centroid.buffer(1000)
            possible_matches_index = list(spatial_index.intersection(geom.bounds))
            possible_matches = iacs.iloc[possible_matches_index].copy()

            inters_df = iacs.loc[iacs.index == index].copy()
            inters_df.geometry = inters_df.geometry.centroid.buffer(1000)
            intersection = gpd.overlay(possible_matches, inters_df, how="intersection", keep_geom_type=False, make_valid=True)
            agri_area_in_buffer = intersection.area.sum()

            ## temp:
            # possible_matches.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\possible_matches.shp")
            # inters_df.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\buffer.shp")
            # intersection.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\intersection.shp")

            intersected_list.append((index, agri_area_in_buffer))
        return intersected_list

    num_processes = 8
    chunk_size = len(iacs) // num_processes
    chunks = [iacs.index[i:i + chunk_size] for i in range(0, len(iacs), chunk_size)]
    results = Parallel(n_jobs=num_processes)(
        delayed(process_chunk)(chunk, iacs)
        for chunk in chunks)
    df_lsts = [pd.DataFrame(sublist, columns=["index", "sumAgr1000"]) for sublist in results]
    intersected_combined = pd.concat(df_lsts)
    intersected_combined.index = intersected_combined["index"]
    intersected_combined.drop(columns="index", inplace=True)
    iacs = pd.merge(iacs, intersected_combined, "left", left_index=True, right_index=True)
    iacs["propAg1000"] = round(iacs["sumAgr1000"] / (math.pi * 1000 * 1000), 2)

    iacs[["field_id", "propAg1000", "sumAgr1000"]].to_csv(r"test_run2\SumAgr\SumAgr_w_grassland.csv", index=False)

    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------

    ###RASTER DATA PROCESSING
    #----------------------------------------------------------------------------------------------------
    #calculate average elevation per field
    with rasterio.open("raster/Dem_3035.tif") as src:
        affine = src.transform #using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array[array == ndval] = np.nan #set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(zonal_stats(iacs, array, all_touched=True, nodata=np.nan, affine=affine, stats=["mean", "std"]))

    zonal_stats_fields.rename(columns={'mean': 'ElevationA', 'std': 'sdElev0'}, inplace=True)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)
    iacs[["field_id", "ElevationA", "sdElev0"]].to_csv(r"test_run2\elevationAvrg\elevationAvrg_w_grassland.csv", index=False)

    #adding field statistics back to original GeoDataFrame
    # dem = pd.concat([iacs, zonal_stats_fields], axis=1)
    # dem.rename(columns={'mean': 'ElevationA'}, inplace=True)
    # dem.rename(columns={'std': 'sdElev0'}, inplace=True)
    # dem.to_file("test_run2/elevationAvrg", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # calculate average TRI per field
    with rasterio.open("raster/DEM_TRI_GER_25m.tif") as src:
        affine = src.transform  # using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array[array == ndval] = np.nan  # set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(
            zonal_stats(iacs, array, all_touched=True, nodata=np.nan, affine=affine, stats=["mean"]))
        zonal_stats_buff500 = pd.DataFrame(
            zonal_stats(iacs.geometry.centroid.buffer(500), array, all_touched=True, nodata=np.nan, affine=affine,
                        stats=["mean"]))
        zonal_stats_buff1000 = pd.DataFrame(
            zonal_stats(iacs.geometry.centroid.buffer(1000), array, all_touched=True, nodata=np.nan, affine=affine,
                        stats=["mean"]))

    ## temporary:
    # iacs = inv_3035.copy()
    # del inv_3035
    # iacs.drop(columns=["hexaSizeM2", "avgTRI1000"], inplace=True)
    # iacs = pd.concat([iacs, zonal_stats_buff1000], axis=1)
    # iacs.rename(columns={"NumFrmFlds": "fieldCount", "mean": "avgTRI1000"}, inplace=True)

    zonal_stats_fields.rename(columns={'mean': 'avgTRI0'}, inplace=True)
    zonal_stats_buff500.rename(columns={'mean': 'avgTRI500'}, inplace=True)
    zonal_stats_buff1000.rename(columns={'mean': 'avgTRI1000'}, inplace=True)
    zonal_stats_fields = pd.concat([zonal_stats_fields, zonal_stats_buff500, zonal_stats_buff1000], axis=1)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)
    iacs[["field_id", "avgTRI0", "avgTRI500", "avgTRI1000"]].to_csv(r"test_run2\TRI\TRI_w_grassland.csv", index=False)

    # adding field statistics back to original GeoDataFrame
    # tri = pd.concat([iacs, zonal_stats_fields, zonal_stats_buff500, zonal_stats_buff1000], axis=1)
    # tri.to_file("test_run2/TRIaverage", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # calculate SQR
    with rasterio.open("raster/sqr1000_250_v10_3035.tif") as src:
        affine = src.transform  # using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array = array.astype('float64')
        array[array == ndval] = np.nan  # set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(
            zonal_stats(iacs, array, affine=affine, nodata=np.nan, all_touched=True, stats=["mean"]))

    zonal_stats_fields.rename(columns={'mean': 'SQRAvrg'}, inplace=True)
    zonal_stats_fields.to_csv(r"test_run2\SQRAvrg\SQRAvrg_w_grassland.csv", index=False)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)
    iacs[["field_id", "SQRAvrg"]].to_csv(r"test_run2\SQRAvrg\SQRAvrg_w_grassland.csv", index=False)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # OUTPUT
    # write shapefile
    iacs.to_file("test_run2/all_predictors_w_grassland", driver="ESRI Shapefile")

    # adding statistics back to original GeoDataFrame
    # sqr = pd.concat([iacs, zonal_stats_fields], axis=1)
    # sqr.rename(columns={'mean': 'SQRAvrg'}, inplace=True)
    # sqr.to_file("test_run2/SQRAvrg", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # calculate average slope per field
    # with rasterio.open("raster/Slope_3035.tif") as src:
    #     affine = src.transform #using affine transformation matrix
    #     array = src.read(1)
    #     ndval = src.nodatavals[0]
    #     #array = array.astype('float64')
    #     array[array==ndval] = np.nan #set no data values to appropriate na-value format
    #     zonal_stats_fields= pd.DataFrame(zonal_stats(iacs, array, all_touched = True, nodata=np.nan, affine=affine, stats=["mean"]))
    #
    # #slope = zonal_stats_alt(iacs, data_in+"raster/Slope_3035.tif")
    #
    # #adding statistics back to original GeoDataFrame
    # slope = pd.concat([iacs, df_zonal_stats], axis=1)
    #
    # #rename columns
    # slope.rename(columns={'mean': 'SlopeAvrg'}, inplace=True)
    #
    # #write shapefile
    # slope.to_file("test_run2/slopeAvrg", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # determine most common crop sequence type per field
    # with rasterio.open("raster/CST/all_2012-2018_CropSeqType_3035.tif") as src:
    #     affine = src.transform #using affine transformation matrix
    #     array = src.read(1)
    #
    #     #ndval = src.nodatavals[0]
    #     #array = array.astype('float')
    #     #array[array==ndval] = np.nan #set no data values to appropriate na-value format
    #     zonal_stats_fields= pd.DataFrame(zonal_stats(iacs, array,nodata=255.0, all_touched=True, affine=affine, stats=["majority"]))
    #
    # # adding statistics back to original GeoDataFrame
    # cst = pd.concat([iacs, df_zonal_stats], axis=1)
    #
    # #rename columns
    # cst.rename(columns={'majority': 'CstMaj'}, inplace=True)
    #
    # #write shapefile
    # cst.to_file("test_run2/CstMaj", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # calculate average field usage capacity (nutzbare Feldkapazitaet NFKW)
    # with rasterio.open("raster/NFKWe1000_250_3035.tif") as src:
    #     affine = src.transform #using affine transformation matrix
    #     array = src.read(1)
    #     ndval = src.nodatavals[0]
    #     array = array.astype('float64')
    #     array[array==ndval] = np.nan #set no data values to appropriate na-value format
    #     zonal_stats_fields= pd.DataFrame(zonal_stats(iacs, array, affine=affine, nodata=np.nan, all_touched = True, stats=["mean"]))
    #
    # #adding statistics back to original GeoDataFrame
    # nfk = pd.concat([iacs, df_zonal_stats], axis=1)

    # #rename columns
    # nfk.rename(columns={'mean': 'NfkAvrg'}, inplace=True)
    #
    # #write shapefile
    # nfk.to_file("test_run2/NfkAvrg", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    #calculate average soil water (Austausch Bodenwasser AHAACGL)
    # with rasterio.open("raster/AHACGL1000_250_3035.tif") as src:
    #     affine = src.transform #using affine transformation matrix
    #     array = src.read(1)
    #     ndval = src.nodatavals[0]
    #     #array = array.astype('float64')
    #     array[array==ndval] = np.nan #set no data values to appropriate na-value format
    #
    #     zonal_stats_fields= pd.DataFrame(zonal_stats(iacs, array, affine=affine, all_touched = True, nodata=np.nan, stats=["mean"]))
    #
    # #adding statistics back to original GeoDataFrame
    # aha = pd.concat([iacs, df_zonal_stats], axis=1)
    #
    # #rename columns
    # aha.rename(columns={'mean': 'AhaacglAvr'}, inplace=True)
    #
    # #write shapefile
    # aha.to_file("test_run2/AhaacglAvrg", driver="ESRI Shapefile")
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------

    #in preparation for the final merge, we first have to get rid of the NAs in field_id, which were produced during the last steps when calculating the raster statistics per field (raster values where were no field geometries were also put in the df and therefore there are a lot of NA values in field_id)

    #check for nas

    #drop nas
    # dem_new = dem.dropna(subset=['field_id'])
    # sqr_new = sqr.dropna(subset=['field_id'])

    # slope_new = slope.dropna(subset = ['field_id'])
    # cst_new = cst.dropna(subset = ['field_id'])
    # nfk_new = nfk.dropna(subset = ['field_id'])
    # aha_new = aha.dropna(subset = ['field_id'])

    #merge
    # merge1 = pd.merge(slope_new, cst_new[["field_id", "CstMaj"]], how="left", on="field_id", validate ="one_to_one")
    # merge2 = pd.merge(nfk_new, aha_new[["field_id", "AhaacglAvr"]], how="left", on="field_id", validate ="one_to_one")
    # merge22 = pd.merge(merge2, dem_new[["field_id", "ElevationA", "sdElev0"]], how="left", on="field_id", validate ="one_to_one")
    # merge33 = pd.merge(merge22, sqr_new[["field_id", "SQRAvrg"]], how="left", on="field_id", validate ="one_to_one")
    # final_merge = pd.merge(merge1, merge33[["field_id", "NfkAvrg", "AhaacglAvr","ElevationA", "SQRAvrg", "sdElev0"]], how="left", on="field_id", validate ="one_to_one")
    # final_merge = pd.merge(final_merge, dem2[[ "field_id", "sdElev500"]], how="left", on="field_id")
    # final_merge_2 = pd.merge(final_merge, dem3[[ "field_id", "sdElev1000"]], how="left", on="field_id")

    # merge = pd.merge(sqr_new[["field_id", "SQRAvrg"]], dem_new[["field_id", "ElevationA", "sdElev0"]], how="left", on="field_id", validate ="one_to_one")
    # final_merge = pd.merge(final_merge, dem2[["hexa_id", "sdElevatio"]], how="left", on="hexa_id")

    #move geometry column to the end of df
    #column_to_move = final_merge.pop("geometry")
    # insert column with insert(location, column_name, column_value)
    #final_merge.insert(11, "geometry", column_to_move)
    #print(final_merge)

    #write shapefile
    # iacs.to_file("test_run2/all_predictors", driver="ESRI Shapefile")

    #----------------------------------------------------------------------------------------------------

    # #do 10% subset of whole dataframe
    # inv_10perc = final_merge.sample(frac= 0.1)
    #
    # #write this subset to csv
    # inv_10perc.to_csv("test_run2/10perc_final_inv.csv", index=False)
    #
    # #write complete dataset to shapefile
    # final_merge.to_file("test_run2/final_inv_compl", driver="ESRI Shapefile")
    #
    #
