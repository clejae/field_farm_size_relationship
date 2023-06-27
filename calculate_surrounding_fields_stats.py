## Authors: 
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as lr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from prop_match_functions import *
import numpy as np

# project library for plotting
import plotting_lib
## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"
IACS_PTH = r"data\test_run2\all_predictors\all_predictors.shp"
SAMPLE_PTH = r"data\vector\final\matched_sample-v2.shp"

## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#

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


## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    sub = gpd.read_file(r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\matched_sample_v1.shp")
    sub_ids = sub["field_id"].tolist()

    calculate_statistics_of_surrounding_fields(
        iacs_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\IACS\IACS_ALL_2018_NEW_3035.shp",
        out_pth=r"Q:\FORLand\Field_farm_size_relationship\data\tables\surrounding_fields_stats_sample.csv",
        sub_ids=sub_ids,
        buffer_radius=1000)

    calculate_statistics_of_surrounding_fields(
        iacs_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\IACS\IACS_ALL_2018_NEW_3035.shp",
        out_pth=r"Q:\FORLand\Field_farm_size_relationship\data\tables\surrounding_fields_stats_ALL.csv",
        buffer_radius=1000)

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
