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
import time
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

def calculate_statistics_of_surrounding_fields(iacs_pth, out_pth, sub_ids=None, buffer_radius=1000):
    """
    1. Calculate centroids, keep field size column.
    2. Draw a buffer around each field (use only subset if provided).
    3. Get centroids that fall in buffer.
    4. Drop current centroid.
    5. Calculate mean, std, median, no. fields.
    :param iacs_pth:
    :param sub_ids:
    :param out_pth:
    :param buffer_radius
    :return:
    """

    print("Read IACS data.")
    iacs = gpd.read_file(iacs_pth)

    print("Calculate the statistics.")
    ## Replace field geometries with centroids, because all further calculation is only based on them
    iacs["geometry"] = iacs["geometry"].centroid

    ## Create a second layer with the buffers
    iacs_buffer = iacs.copy()
    iacs_buffer["geometry"] = iacs_buffer["geometry"].buffer(buffer_radius)
    iacs_buffer = iacs_buffer[["field_id", "geometry"]]

    ## Get field ids for which the calculation should be done
    if not sub_ids:
        field_ids = list(iacs["field_id"].unique())
    else:
        field_ids = sub_ids

    ## Intersect buffers with centroids
    intersection = gpd.sjoin(iacs_buffer.loc[iacs_buffer["field_id"].isin(field_ids)], iacs)

    ## For all buffers, drop fields that are the centre of the buffers, because they should not impact the statistics
    intersection = intersection.loc[intersection["field_id_left"] != intersection["field_id_right"]].copy()

    ## Calculate the statistics
    stats = intersection.groupby("field_id_left").agg(
        surrf_mean=pd.NamedAgg("field_size", "mean"),
        surrf_median=pd.NamedAgg("field_size", "median"),
        surrf_std=pd.NamedAgg("field_size", "std"),
        surrf_min=pd.NamedAgg("field_size", "min"),
        surrf_max=pd.NamedAgg("field_size", "max"),
        surrf_no_fields=pd.NamedAgg("field_id_right", "nunique")
    ).reset_index()

    stats.rename(columns={"field_id_left": "field_id"}, inplace=True)

    print("Write results to:", out_pth)
    stats.to_csv(out_pth, index=False)

    print("Done!")

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


def get_size_and_administrative_variables(iacs_pth, size_vars_out_pth):
    # ----------------------------------------------------------------------------------------------------
    ###VECTOR DATA PROCESSING
    ##read vector data
    # invekos
    iacs = gpd.read_file(iacs_pth)
    iacs.dropna(inplace=True)

    # data info
    print("CRS IACS:", iacs.crs)

    # ----------------------------------------------------------------------------------------------------
    # calculate field sizes and hexagon sizes

    # calculate field size
    iacs["fieldSizeM2"] = iacs.area

    # ----------------------------------------------------------------------------------------------------
    ###Count number of fields per farm
    df = iacs.groupby(['farm_id']).agg(
        fieldCount=pd.NamedAgg("field_id", "count")
    ).reset_index()

    # merge to invekos dataframe
    iacs = pd.merge(iacs, df, how="left", on="farm_id")

    # ----------------------------------------------------------------------------------------------------
    ###assign if former East or West Germany member
    iacs.rename(columns={"state": "federal_state"}, inplace=True)
    iacs['FormerDDRmember'] = 0
    iacs.loc[iacs["federal_state"].isin(["BB", "SA", "TH"]), "FormerDDRmember"] = 1
    print("Values for FormerDDRmember:", iacs.FormerDDRmember.unique())

    iacs.drop(columns=["geometry"], inplace=True)
    iacs.to_csv(size_vars_out_pth, index=False)


def calculate_proportion_agricultural_area(iacs_pth, sum_agr_out_pth):

    print("Read IACS")
    iacs = gpd.read_file(iacs_pth)
    iacs.dropna(inplace=True)
    iacs.index = range(len(iacs))
    def process_chunk(chunk, c, iacs):
        print("Processing chunk", c)
        ## temp:
        # chunk = iacs.index[0:0 + 10]
        # index = chunk[1]
        ## creat spatial index
        spatial_index = iacs.copy().sindex
        intersected_list = []
        ## loop over rows
        for i in chunk:
            # print(i)
            row = iacs.iloc[i].copy()
            # field_id = iacs["field_id"].iloc[i]
            geom = row.geometry.centroid.buffer(1000)
            possible_matches_index = list(spatial_index.intersection(geom.bounds))
            possible_matches = iacs.iloc[possible_matches_index].copy()
            inters_df = iacs.loc[iacs.index == i].copy()
            inters_df.geometry = inters_df.geometry.centroid.buffer(1000)
            try:
                intersection = gpd.overlay(possible_matches, inters_df, how="intersection", keep_geom_type=False,
                                           make_valid=True)
                agri_area_in_buffer = intersection.area.sum()
            except:
                agri_area_in_buffer = 0

            # possible_matches.to_file(
            #     fr"Q:\FORLand\Field_farm_size_relationship\data\tables\predictors\possible_matches_{index}.gpkg",
            #     driver="GPKG")
            # inters_df.to_file(rf"Q:\FORLand\Field_farm_size_relationship\data\tables\predictors\inters_df_{index}.gpkg",
            #                   driver="GPKG")
            ## temp:
            # possible_matches.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\possible_matches.shp")
            # inters_df.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\buffer.shp")
            # intersection.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\intersection.shp")

            intersected_list.append((i, agri_area_in_buffer))

        df_res = pd.DataFrame(intersected_list, columns=["index", "sumAgr1000"])
        df_res.to_csv(sum_agr_out_pth[:-4] + f'_chunk{c}.csv', index=False)

        return intersected_list

    num_processes = 10
    chunk_size = len(iacs) // num_processes
    chunks = [iacs.index[i:i + chunk_size] for i in range(0, len(iacs), chunk_size)]

    results = Parallel(n_jobs=num_processes)(
        delayed(process_chunk)(chunk, c, iacs)
        for c, chunk in enumerate(chunks))

    # ## Because parallel processing doesn't work, I wrote a loop
    # spatial_index = iacs.copy().sindex
    # results = []
    # for x, chunk in enumerate(chunks):
    #     print("Processing", x, "chunk of", len(chunks), "chunks.")
    #     intersected_list = []
    #     for i in chunk:
    #         # print(i)
    #         row = iacs.iloc[i].copy()
    #         # field_id = iacs["field_id"].iloc[i]
    #         geom = row.geometry.centroid.buffer(1000)
    #         possible_matches_index = list(spatial_index.intersection(geom.bounds))
    #         possible_matches = iacs.iloc[possible_matches_index].copy()
    #         inters_df = iacs.loc[iacs.index == i].copy()
    #         inters_df.geometry = inters_df.geometry.centroid.buffer(1000)
    #         try:
    #             intersection = gpd.overlay(possible_matches, inters_df, how="intersection", keep_geom_type=False, make_valid=True)
    #             agri_area_in_buffer = intersection.area.sum()
    #         except:
    #             agri_area_in_buffer = 0
    #
    #         # possible_matches.to_file(
    #         #     fr"Q:\FORLand\Field_farm_size_relationship\data\tables\predictors\possible_matches_{index}.gpkg",
    #         #     driver="GPKG")
    #         # inters_df.to_file(rf"Q:\FORLand\Field_farm_size_relationship\data\tables\predictors\inters_df_{index}.gpkg",
    #         #                   driver="GPKG")
    #         ## temp:
    #         # possible_matches.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\possible_matches.shp")
    #         # inters_df.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\buffer.shp")
    #         # intersection.to_file(r"Q:\FORLand\Field_farm_size_relationship\temp\intersection.shp")
    #
    #         intersected_list.append((i, agri_area_in_buffer))
    #
    #     df_res = pd.DataFrame(intersected_list, columns=["index", "sumAgr1000"])
    #     df_res.to_csv(sum_agr_out_pth[:-4] + f'_chunk{x}.csv', index=False)
    #     results.append(intersected_list)

    df_lsts = [pd.DataFrame(sublist, columns=["index", "sumAgr1000"]) for sublist in results]
    intersected_combined = pd.concat(df_lsts)
    intersected_combined.index = intersected_combined["index"]
    intersected_combined.drop(columns="index", inplace=True)
    iacs = pd.merge(iacs, intersected_combined, "left", left_index=True, right_index=True)
    iacs["propAg1000"] = round(iacs["sumAgr1000"] / (math.pi * 1000 * 1000), 2)

    iacs[["field_id", "propAg1000", "sumAgr1000"]].to_csv(sum_agr_out_pth, index=False)


def calculate_elevation_per_field(iacs_pth, dem_pth, elevation_out_pth):

    print("Read IACS")
    iacs = gpd.read_file(iacs_pth)
    iacs.dropna(inplace=True)

    with rasterio.open(dem_pth) as src:
        affine = src.transform  # using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array[array == ndval] = np.nan  # set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(
            zonal_stats(iacs, array, all_touched=True, nodata=np.nan, affine=affine, stats=["mean", "std"]))

    zonal_stats_fields.rename(columns={'mean': 'ElevationA', 'std': 'sdElev0'}, inplace=True)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)

    iacs[["field_id", "ElevationA", "sdElev0"]].to_csv(elevation_out_pth, index=False)


def calculate_terrain_ruggedness_index_per_field(iacs_pth, tri_pth, tri_out_pth):
    print("Read IACS")
    iacs = gpd.read_file(iacs_pth)
    iacs.dropna(inplace=True)

    with rasterio.open(tri_pth) as src:
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

    zonal_stats_fields.rename(columns={'mean': 'avgTRI0'}, inplace=True)
    zonal_stats_buff500.rename(columns={'mean': 'avgTRI500'}, inplace=True)
    zonal_stats_buff1000.rename(columns={'mean': 'avgTRI1000'}, inplace=True)
    zonal_stats_fields = pd.concat([zonal_stats_fields, zonal_stats_buff500, zonal_stats_buff1000], axis=1)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)

    iacs[["field_id", "avgTRI0", "avgTRI500", "avgTRI1000"]].to_csv(tri_out_pth, index=False)


def calculate_soil_quality_per_field(iacs_pth, sqr_pth, sqr_out_pth):
    print("Read IACS")
    iacs = gpd.read_file(iacs_pth)
    iacs.dropna(inplace=True)

    with rasterio.open(sqr_pth) as src:
        affine = src.transform  # using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array = array.astype('float64')
        array[array == ndval] = np.nan  # set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(
            zonal_stats(iacs, array, affine=affine, nodata=np.nan, all_touched=True, stats=["mean"]))

    zonal_stats_fields.rename(columns={'mean': 'SQRAvrg'}, inplace=True)
    # zonal_stats_fields.to_csv(r"test_run2\SQRAvrg\SQRAvrg_w_grassland.csv", index=False)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)
    iacs[["field_id", "SQRAvrg"]].to_csv(sqr_out_pth, index=False)


def concatenate_predictors(size_vars_pth, sum_agr_pth, elevation_pth, tri_pth, sqr_pth, surrf_pth, predictors_out_pth):

    sizes = pd.read_csv(size_vars_pth)
    sumagr = pd.read_csv(sum_agr_pth)
    elev = pd.read_csv(elevation_pth)
    tri = pd.read_csv(tri_pth)
    sqr = pd.read_csv(sqr_pth)
    surr = pd.read_csv(surrf_pth)

    for col in surr.columns:
        surr.loc[surr[col].isna(), col] = 0
    print(len(surr), len(surr.loc[surr.isnull().any(axis=1)]))

    df_out = pd.merge(sizes, sumagr, "left", "field_id")
    df_out = pd.merge(df_out, elev, "left", "field_id")
    df_out = pd.merge(df_out, tri, "left", "field_id")
    df_out = pd.merge(df_out, sqr, "left", "field_id")
    df_out = pd.merge(df_out, surr, "left", "field_id")

    for col in surr.columns:
        df_out.loc[df_out[col].isna(), col] = 0

    for col in ["avgTRI0", "ElevationA"]:
        df_out = df_out.loc[~df_out[col].isna()].copy()

    df_out.to_csv(predictors_out_pth, index=False)

def prepare_explanatory_variables(iacs_pth, dem_pth, tri_pth, sqr_pth, sum_agr_out_pth, elevation_out_pth,
                                  tri_out_pth, sqr_out_pth, predictors_out_pth):
    # ----------------------------------------------------------------------------------------------------
    ###VECTOR DATA PROCESSING
    ##read vector data
    # invekos
    iacs = gpd.read_file(iacs_pth)
    iacs.dropna(inplace=True)

    # data info
    print("CRS IACS:", iacs.crs)

    # ----------------------------------------------------------------------------------------------------
    # calculate field sizes and hexagon sizes

    # calculate field size
    iacs["fieldSizeM2"] = iacs.area
    # iacs.drop(["field_size"], axis=1, inplace=True)

    # # calculate area size per hexagon
    # hexagon grid
    # hexa = gpd.read_file(hexa_pth)
    # hexa["hexaSizeM2"] = hexa.area
    #
    # # rename id to hex_id for clarity
    # hexa.rename(columns={'id': 'hexa_id'}, inplace=True)

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

    # # calculate centroids of fields
    # iacs["centroids"] = iacs["geometry"].centroid
    # centroids = iacs[["field_id", "centroids"]].copy()
    # centroids.rename(columns={"centroids": "geometry"}, inplace=True)
    # centroids = centroids.set_geometry("geometry")
    # iacs.drop(columns=["centroids"], inplace=True)
    #
    # ## Create a centroids layer and add the hex id to the centroids
    # centroids = gpd.GeoDataFrame(centroids, crs=25832)
    # centroids = gpd.sjoin(centroids, hexa, how="left", op="intersects")
    #
    # # add centroids info to invekos dataframe
    # # check if same length
    #
    # if len(iacs) == len(centroids):
    #     print("Merging IACS with centroids and hexagon information.")
    #     iacs = pd.merge(iacs, centroids[["field_id", "hexa_id"]], how="left", on="field_id")
    # else:
    #     warnings.warn("IACS and centroids don't have the same length!", len(iacs), len(centroids))

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

    num_processes = 20
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

    iacs[["field_id", "propAg1000", "sumAgr1000"]].to_csv(sum_agr_out_pth, index=False)

    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------

    ###RASTER DATA PROCESSING
    #----------------------------------------------------------------------------------------------------
    #calculate average elevation per field

    with rasterio.open(dem_pth) as src:
        affine = src.transform #using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array[array == ndval] = np.nan #set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(zonal_stats(iacs, array, all_touched=True, nodata=np.nan, affine=affine, stats=["mean", "std"]))

    zonal_stats_fields.rename(columns={'mean': 'ElevationA', 'std': 'sdElev0'}, inplace=True)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)

    iacs[["field_id", "ElevationA", "sdElev0"]].to_csv(elevation_out_pth, index=False)

    #adding field statistics back to original GeoDataFrame
    # dem = pd.concat([iacs, zonal_stats_fields], axis=1)
    # dem.rename(columns={'mean': 'ElevationA'}, inplace=True)
    # dem.rename(columns={'std': 'sdElev0'}, inplace=True)
    # dem.to_file("test_run2/elevationAvrg", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # calculate average TRI per field

    with rasterio.open(tri_pth) as src:
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

    iacs[["field_id", "avgTRI0", "avgTRI500", "avgTRI1000"]].to_csv(tri_out_pth, index=False)

    # adding field statistics back to original GeoDataFrame
    # tri = pd.concat([iacs, zonal_stats_fields, zonal_stats_buff500, zonal_stats_buff1000], axis=1)
    # tri.to_file("test_run2/TRIaverage", driver="ESRI Shapefile")

    # ----------------------------------------------------------------------------------------------------
    # calculate SQR

    with rasterio.open(sqr_pth) as src:
        affine = src.transform  # using affine transformation matrix
        array = src.read(1)
        ndval = src.nodatavals[0]
        array = array.astype('float64')
        array[array == ndval] = np.nan  # set no data values to appropriate na-value format
        zonal_stats_fields = pd.DataFrame(
            zonal_stats(iacs, array, affine=affine, nodata=np.nan, all_touched=True, stats=["mean"]))

    zonal_stats_fields.rename(columns={'mean': 'SQRAvrg'}, inplace=True)
    # zonal_stats_fields.to_csv(r"test_run2\SQRAvrg\SQRAvrg_w_grassland.csv", index=False)
    iacs = pd.concat([iacs, zonal_stats_fields], axis=1)
    iacs[["field_id", "SQRAvrg"]].to_csv(sqr_out_pth, index=False)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # OUTPUT
    # write shapefile
    if predictors_out_pth.endswith('.gpkg'):
        iacs.to_file(predictors_out_pth, driver="GPKG")
    elif predictors_out_pth.endswith('.csv'):
        iacs.drop(columns=["geometry"], inplace=True)
        iacs.to_csv(predictors_out_pth, index=False)

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


def merge_surr_fields_with_other_variables(iacs_pth, surrf_pth, out_pth):
    if iacs_pth.endswith('.gpkg') | iacs_pth.endswith('.shp'):
        iacs = gpd.read_file(iacs_pth)
    elif iacs_pth.endswith('.csv'):
        iacs = pd.read_csv(iacs_pth)
    else:
        return

    surr = pd.read_csv(surrf_pth)
    for col in surr.columns:
        surr.loc[surr[col].isna(), col] = 0
    print(len(surr), len(surr.loc[surr.isnull().any(axis=1)]))
    iacs = pd.merge(iacs, surr, "left", "field_id")
    for col in surr.columns:
        iacs.loc[iacs[col].isna(), col] = 0

    # iacs.to_file(out_pth)
    # iacs.drop(columns="geometry", inplace=True)
    iacs.to_csv(out_pth[:-3] + "csv", index=False)


def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    calculate_statistics_of_surrounding_fields(
        iacs_pth=r"vector\IACS\IACS_ALL_2018_cleaned.gpkg",
        out_pth=r"tables\predictors\surrounding_fields_stats_ALL_1000m.csv",
        buffer_radius=1000)

    get_size_and_administrative_variables(
        iacs_pth=r"vector/IACS/IACS_ALL_2018_cleaned.gpkg",
        size_vars_out_pth="tables/predictors/size_vars_w_grassland.csv")

    calculate_proportion_agricultural_area(
        iacs_pth=r"vector/IACS/IACS_ALL_2018_cleaned.gpkg",
        sum_agr_out_pth="tables/predictors/SumAgr_w_grassland_ALL.csv")

    calculate_elevation_per_field(
        iacs_pth=r"vector/IACS/IACS_ALL_2018_cleaned.gpkg",
        dem_pth="raster/Dem_3035.tif",
        elevation_out_pth=r"tables/predictors/elevationAvrg_w_grassland.csv")

    calculate_terrain_ruggedness_index_per_field(
        iacs_pth=r"vector/IACS/IACS_ALL_2018_cleaned.gpkg",
        tri_pth="raster/DEM_TRI_GER_25m.tif",
        tri_out_pth=r"tables/predictors/TRI_w_grassland.csv")

    calculate_soil_quality_per_field(
        iacs_pth=r"vector/IACS/IACS_ALL_2018_cleaned.gpkg",
        sqr_pth="raster/sqr1000_250_v10_3035.tif",
        sqr_out_pth=r"tables/predictors/SQRAvrg_w_grassland.csv")

    concatenate_predictors(
        size_vars_pth="tables/predictors/size_vars_w_grassland.csv",
        sum_agr_pth="tables/predictors/SumAgr_w_grassland_ALL.csv",
        elevation_pth=r"tables/predictors/elevationAvrg_w_grassland.csv",
        tri_pth=r"tables/predictors/TRI_w_grassland.csv",
        sqr_pth=r"tables/predictors/SQRAvrg_w_grassland.csv",
        surrf_pth=r"tables/predictors/surrounding_fields_stats_ALL_1000m.csv",
        predictors_out_pth="tables/predictors/all_predictors_w_grassland.csv"
    )


    # # prepare_explanatory_variables(
    # #     iacs_pth=r"vector/IACS/IACS_ALL_2018_cleaned.gpkg",
    # #     dem_pth="raster/Dem_3035.tif",
    # #     tri_pth="raster/DEM_TRI_GER_25m.tif",
    # #     sqr_pth="raster/sqr1000_250_v10_3035.tif",
    # #     sum_agr_out_pth="tables/predictors/SumAgr_w_grassland.csv",
    # #     elevation_out_pth=r"tables/predictors/elevationAvrg_w_grassland.csv",
    # #     tri_out_pth=r"tables/predictors/TRI_w_grassland.csv",
    # #     sqr_out_pth=r"tables/predictors/SQRAvrg_w_grassland.csv",
    # #     predictors_out_pth="tables/predictors/soil_landscape_predictors.csv"
    # # )
    #
    # # merge_surr_fields_with_other_variables(
    # #     iacs_pth=r"tables/predictors/soil_landscape_predictors.csv",
    # #     surrf_pth=r"tables/predictors/surrounding_fields_stats_ALL_1000m.csv",
    # #     out_pth=r"tables/predictors/all_predictors_w_grassland.csv"
    # # )

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)

if __name__ == '__main__':
    main()

