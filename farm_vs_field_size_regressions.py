## Authors: 
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import os
import time
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from osgeo import gdal

# project library for plotting
import plotting_lib
## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"


## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#


def prepare_large_iacs_shp(input_dict, out_pth, grassland=True):
    print("Join all IACS files together.")

    shp_lst = []
    for key in input_dict:
        print(f"\tProcessing IACS data of {key}")
        iacs_pth = input_dict[key]["iacs_pth"]
        farm_id_col = input_dict[key]["farm_id_col"]
        crop_name_col = input_dict[key]["crop_name_col"]

        shp = gpd.read_file(iacs_pth)
        shp["field_id"] = shp.index
        shp["field_id"] = shp["field_id"].apply(lambda x: f"{key}_{x}")
        if grassland == False:
            if "ID_KTYP" in list(shp.columns):
                shp["ID_KTYP"] = shp["ID_KTYP"].astype(int)
                shp = shp.loc[shp["ID_KTYP"].isin([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 60])].copy()
        shp = shp.loc[shp["geometry"] != None].copy()

        shp["field_size"] = shp["geometry"].area / 10000

        shp.rename(columns={farm_id_col: "farm_id"}, inplace=True)
        shp.rename(columns={crop_name_col: "crop_name"}, inplace=True)
        shp = shp[["field_id", "farm_id", "field_size", "crop_name", "ID_KTYP", "geometry"]].copy()

        shp_lst.append(shp)

    print("\tConcatenating all shapefiles and writing out.")
    out_shp = pd.concat(shp_lst, axis=0)
    del shp_lst

    ## Calculate farm sizes
    farm_sizes = out_shp.groupby("farm_id").agg(
        farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
    ).reset_index()
    out_shp = pd.merge(out_shp, farm_sizes, on="farm_id", how="left")
    out_shp = out_shp[["field_id", "farm_id", "field_size", "farm_size", "crop_name", "ID_KTYP", "geometry"]].copy()
    # t = out_shp.loc[out_shp["geometry"] != None].copy()
    out_shp.to_crs(3035)
    out_shp.to_file(out_pth)

    ## Temporary:
    if grassland == True:
        out_shp["state"] = out_shp["field_id"].apply(lambda x: x.split("_")[0])
        unique_crop_names = out_shp.loc[out_shp["ID_KTYP"].isin([30, 99])].copy()
        unique_crop_names = unique_crop_names[["state", "crop_name"]].drop_duplicates()
        unique_crop_names.to_csv(r"data\tables\unique_crop_names.csv")

    print("done!")
    return out_shp


def get_unqiue_crop_names_and_crop_classes_relations(iacs, crop_name_col, crop_class_col, out_pth):
    print("Get unique crop name - crop class relations for manual reclassification.")
    iacs.drop_duplicates(subset=[crop_name_col, crop_class_col], inplace=True)
    out_df = iacs[[crop_name_col, crop_class_col]]
    out_df.to_csv(out_pth, sep=";", index=False)


def new_crop_names_classification(iacs, crop_name_col, crop_class_col, classification_df, new_crop_class_col, out_pth,
                                  validity_check_pth=None):
    print("Assgin new crop classes.")
    iacs.drop(columns=[crop_class_col], inplace=True)

    classification = dict(zip(classification_df[crop_name_col], classification_df[new_crop_class_col]))
    iacs[crop_class_col] = iacs[crop_name_col].map(classification)

    ## Translate crop codes to crop class names
    t_dict = {
        1: "MA", #maize
        2: "WW", #winter wheat
        3: "SU", #sugar beet
        4: "OR", #oilseed rape
        5: "PO", #potato
        6: "SC", #summer cereals
        7: "TR", #triticale
        9: "WB", #winter barely
        10: "WR", #winter rye
        12: "LE", #legumes
        13: "AR", #arable grass
        14: "LE", #legumes
        20: "GR", #permanent grassland
        30: "FA", #unkown
        40: "PE", #40-mehrjährige Kulturen und Dauerkulturen
        60: "LE", #legumes
        70: "FA", #field stripes
        80: "UN", #unkown
        90: "GA", #garden flowers
        95: "MI", #mix
        99: "FA", #fallow
    }



    iacs[crop_class_col] = iacs[crop_class_col].map(t_dict)
    iacs[crop_class_col].unique()
    iacs = iacs[iacs[crop_class_col] != "UN"].copy()

    iacs.to_file(out_pth)

    if validity_check_pth:
        out_df = iacs[[crop_name_col, crop_class_col]].copy()
        out_df.drop_duplicates(inplace=True)
        out_df.to_csv(validity_check_pth, sep=";", index=False)

    return iacs

def explore_grasslands(iacs, field_id_col, field_size_col, farm_id_col, farm_size_col,  out_pth):
    print("Explore field and farm sizes of grasslands.")
    ## Derive federal state from field id
    federal_state_col = "federal_state"
    iacs[federal_state_col] = iacs[field_id_col].apply(lambda x: x.split("_")[0])

    iacs["grassland"] = 0
    iacs.loc[iacs["ID_KTYP"] == "GR", "grassland"] = 1
    iacs.loc[iacs["ID_KTYP"].isin(["FA", "MI", "PE", "GA"]), "grassland"] = 2

    ## Calculate field statistice for arable fields, grassland, and fields that did not fall into the CSTypology
    field_stats = iacs.groupby(["grassland"]).agg(
        area=pd.NamedAgg(field_size_col, sum),
        mean_field_size=pd.NamedAgg(field_size_col, np.mean),
        std_field_size=pd.NamedAgg(field_size_col, np.std)
    )

    ## Calculate farm statistics for farms with grassland and without
    def calc_grass_share(group):
        total_area = group[field_size_col].sum()
        grass_area = group.loc[group["grassland"] == 1, field_size_col].sum()
        return round(grass_area / total_area, 2)

    grass_share = iacs.groupby([farm_id_col]).apply(calc_grass_share).reset_index()
    grass_share.columns = [farm_id_col, "grass_share"]

    farms = iacs.groupby(farm_id_col).agg(
        farm_size=pd.NamedAgg(farm_size_col, "first"),
        federal_state=pd.NamedAgg(federal_state_col, "first"),
        n_farm_sizes=pd.NamedAgg(farm_size_col, "nunique"),  # to control for consistency
        mean_field_size=pd.NamedAgg(field_size_col, "mean"),
        num_fields=pd.NamedAgg(field_id_col, "nunique")
    ).reset_index()

    farms = pd.merge(farms, grass_share, how="left", on=farm_id_col)
    farms["with_grassland"] = 0
    farms.loc[farms["grass_share"] > 0, "with_grassland"] = 1

    farm_stats = farms.groupby(["with_grassland"]).agg(
        mean_farm_area=pd.NamedAgg("farm_size", "mean"),
        mean_field_size=pd.NamedAgg("mean_field_size", "mean"),
        mean_num_fields=pd.NamedAgg("num_fields", "mean"),
        total_area=pd.NamedAgg("farm_size", "sum"),
        num_farms=pd.NamedAgg(farm_id_col, "nunique")
    ).reset_index()

    farm_stats_fs = farms.groupby(["with_grassland", "federal_state"]).agg(
        mean_farm_area=pd.NamedAgg("farm_size", "mean"),
        mean_field_size=pd.NamedAgg("mean_field_size", "mean"),
        mean_num_fields=pd.NamedAgg("num_fields", "mean"),
        total_area=pd.NamedAgg("farm_size", "sum"),
        num_farms=pd.NamedAgg(farm_id_col, "nunique")
    ).reset_index()

    ## Write out
    with pd.ExcelWriter(out_pth) as writer:
        field_stats.to_excel(writer, sheet_name='field_stats')
        farm_stats.to_excel(writer, sheet_name='farm_stats')
        farm_stats_fs.to_excel(writer, sheet_name='farm_stats_fs')

    print("Done!")


def explore_farm_distribution_across_states(iacs, shp_out_pth, csv_out_pth, plt_out_pth):
    print("Explore farm distribution across states borders.")
    iacs["state"] = iacs["field_id"].apply(lambda x: x.split("_")[0])

    df_agg = iacs.groupby(["farm_id"]).agg(
        num_states=pd.NamedAgg("state", pd.Series.nunique)
    ).reset_index()

    df_agg = df_agg.loc[df_agg["num_states"] > 1].copy()

    iacs_sub = iacs.loc[iacs["farm_id"].isin(list(df_agg["farm_id"]))].copy()
    iacs_sub.to_file(shp_out_pth)

    def calc_state_area(group, state):
        state_area = group.loc[group["state"] == state, "field_size"].sum()
        return state_area

    bb = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='BB').reset_index(name="bb_area")
    sa = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='SA').reset_index(name="sa_area")
    th = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='TH').reset_index(name="th_area")
    ls = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='LS').reset_index(name="ls_area")
    bv = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='BV').reset_index(name="bv_area")

    from functools import reduce
    df_out1 = reduce(lambda left, right: pd.merge(left, right, on='farm_id'), [bb, sa, th, ls, bv])

    def count_fields(group, state):
        num_fields = len(group.loc[group["state"] == state, "field_id"].unique())
        return num_fields

    bb = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='BB').reset_index(name="bb_num_fields")
    sa = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='SA').reset_index(name="sa_num_fields")
    th = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='TH').reset_index(name="th_num_fields")
    ls = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='LS').reset_index(name="ls_num_fields")
    bv = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='BV').reset_index(name="bv_num_fields")

    df_out2 = reduce(lambda left, right: pd.merge(left, right, on='farm_id'), [bb, sa, th, ls, bv])

    df_out = pd.merge(df_out1, df_out2, "left", "farm_id")

    cols = ["bb_area", "sa_area", "th_area", "ls_area", "bv_area"]
    df_out["main_state"] = df_out[cols].idxmax(axis=1)
    df_out["farm_size"] = df_out[cols].sum(axis=1)
    df_out["main_st_area"] = df_out[cols].max(axis=1)
    df_out["out_st_area"] = df_out["farm_size"] - df_out["main_st_area"]
    df_out["share_out"] = round(df_out["out_st_area"] / df_out["farm_size"] * 100, 1)

    cols = ["bb_num_fields", "sa_num_fields", "th_num_fields", "ls_num_fields", "bv_num_fields"]
    df_out["num_fields"] = df_out[cols].sum(axis=1)
    df_out["main_st_fields"] = df_out[cols].max(axis=1)
    df_out["out_st_fields"] = df_out["num_fields"] - df_out["main_st_fields"]

    fig, ax = plt.subplots()
    sns.histplot(data=df_out, x="share_out", bins=30, ax=ax)
    ax.annotate(s=f"Only for BB, SA, TH, as LS and BV\nuse different farm identifiers.", xy=(29, 50))
    ax.annotate(s=f"Total affected area: {round(df_out['out_st_area'].sum(), 0)} ha", xy=(29, 43))
    ax.annotate(s=f"No. affected fields: {round(df_out['out_st_fields'].sum(), 0)}", xy=(29, 40))
    ax.annotate(s=f"No. affected farms: {len(df_out['farm_id'].unique())}", xy=(29, 37))
    ax.set_xlabel("Share of farm outside 'main state' [%]")

    plt.savefig(plt_out_pth)

    df_out.to_csv(csv_out_pth, index=False)


def explore_field_sizes(iacs, farm_id_col, field_id_col, field_size_col, farm_size_col, crop_class_col):
    print("Explore field sizes.")
    ## Derive federal state from field id
    federal_state_col = "federal_state"
    iacs[federal_state_col] = iacs[field_id_col].apply(lambda x: x.split("_")[0])

    ## Derive general statistics about each farm
    farms = iacs.groupby(farm_id_col).agg(
        farm_size=pd.NamedAgg(farm_size_col, "first"),
        n_farm_sizes=pd.NamedAgg(farm_size_col, "nunique"), # to control for consistency
        mean_field_size=pd.NamedAgg(field_size_col, "mean"),
        max_field_size=pd.NamedAgg(field_size_col, "max"),
        min_field_size=pd.NamedAgg(field_size_col, "min"),
        num_fields=pd.NamedAgg(field_id_col, "nunique")
    )

    consistency_test = farms.loc[farms["n_farm_sizes"] > 1].copy()

    if not consistency_test.empty:
        warnings.warn("IACS dataframe has multiple farm sizes for a single farm!")
    else:
        print("Consistency test passed.")

    ## Generate some ratios based on the maximum and minimum field sizes
    farms["range_field_sizes"] = farms["max_field_size"] - farms["min_field_size"]
    farms["range_field_ratio"] = farms["range_field_sizes"] / farms["farm_size"]
    farms["max_field_ratio"] = farms["max_field_size"] / farms["farm_size"]
    farms["min_field_ratio"] = farms["min_field_size"] / farms["farm_size"]
    farms["diff_ratios"] = farms["max_field_ratio"] - farms["min_field_ratio"]

    farms["log_max_field_ratio"] = np.log(farms["max_field_ratio"])
    farms["log_min_field_ratio"] = np.log(farms["min_field_ratio"])
    farms["log_farm_size"] = np.log(farms["farm_size"])

    ## Classify farms based on their total farm size
    farms["farm_size_class"] = pd.cut(farms["farm_size"], [0, 10, 25, 100, 250, 1000, 2500], labels=[10, 25, 100, 250, 1000, 2500])
    farms["farm_size_class"] = pd.qcut(x=farms["farm_size"], q=10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ## Plot farm size vs max and minimum field size ratios in scatterplot
    # fig, axs = plt.subplots(nrows=3, ncols=1)
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, hue="num_fields", ax=axs[0])
    # sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs[1])
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, ax=axs[2])
    # fig.tight_layout()

    ## Plot farm size vs difference of maximum and minimum field size
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="diff_ratios", s=3, ax=axs)
    # fig.tight_layout()

    ## Plot farm size vs range of maximum and minimum field size
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="range_field_ratio", s=3, ax=axs)
    # fig.tight_layout()

    ## Plot farm size vs maximum and minimum field size ratio
    fig, axs = plt.subplots()
    sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, ax=axs)
    sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs)
    axs.set_xlabel("Farm size [ha]")
    axs.set_ylabel("ratio of max. (blue) and min. (orange) field sizes")
    fig.tight_layout()
    plt.savefig(fr"figures\scatterplot_max+min_field_ratios_with_grassland.png")
    plt.close()

    ## Plot minimum and maximum field sizes per farm size class in boxplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=plotting_lib.cm2inch((16, 6)))
    sns.boxplot(data=farms, x="farm_size_class", y="min_field_size", showfliers=False, ax=axs[1])
    axs[1].set_xlabel("Deciles of farm sizes")
    axs[1].set_ylabel("Min. field size [ha]")
    axs[1].set_title("b)", loc="left")
    axs[1].grid(visible=True, which="major", axis="y", zorder=0)
    axs[1].set_axisbelow(True)
    sns.boxplot(data=farms, x="farm_size_class", y="max_field_size", showfliers=False, ax=axs[0])
    axs[0].set_xlabel("Deciles of farm sizes")
    axs[0].set_ylabel("Max. field size [ha]")
    axs[0].set_title("a)", loc="left")
    axs[0].grid(visible=True, which="major", axis="y", zorder=0)
    axs[0].set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(fr"figures\boxplot_min_max_field_sizes_per_farm_decile_with_grassland.png")

    ## Calculate area of crop classes in total and per federal state
    crop_classes = iacs[[crop_class_col, federal_state_col, field_size_col]].groupby([crop_class_col, federal_state_col]).agg(
        total_area=pd.NamedAgg(field_size_col, "sum"),
        mean_area=pd.NamedAgg(field_size_col, "mean"),
        max_area=pd.NamedAgg(field_size_col, "max"),
        min_area=pd.NamedAgg(field_size_col, "min"),
        median_area=pd.NamedAgg(field_size_col, "median")
    ).reset_index()
    crop_classes["number_fields"] = crop_classes["total_area"] / crop_classes["mean_area"]
    crop_classes["share"] = round(crop_classes["total_area"] / crop_classes["total_area"].sum() * 100, 2)
    crop_classes.to_csv(fr"data\tables\statistics_about_crop_classes_per_federal_state.csv", index=False, sep=";")

    crop_classes_total = iacs[[crop_class_col, field_size_col]].groupby([crop_class_col]).agg(
        total_area=pd.NamedAgg(field_size_col, "sum"),
        mean_area=pd.NamedAgg(field_size_col, "mean"),
        max_area=pd.NamedAgg(field_size_col, "max"),
        min_area=pd.NamedAgg(field_size_col, "min"),
        median_area=pd.NamedAgg(field_size_col, "median")
    ).reset_index()
    crop_classes_total["number_fields"] = crop_classes_total["total_area"] / crop_classes_total["mean_area"]
    crop_classes_total["share"] = round(crop_classes_total["total_area"] / crop_classes_total["total_area"].sum() * 100, 2)
    crop_classes_total.to_csv(fr"data\tables\statistics_about_crop_classes.csv", index=False, sep=";")

    fig, axs = plt.subplots(nrows=2, figsize=plotting_lib.cm2inch((16, 12)))
    sns.boxplot(data=iacs, x=crop_class_col, y=field_size_col, showfliers=False, ax=axs[0])
    axs[0].set_xlabel("Crop class")
    axs[0].set_ylabel("Field size")
    axs[0].set_title("a) Field sizes", loc="left")
    axs[0].grid(visible=True, which="major", axis="y", zorder=0)
    axs[0].set_axisbelow(True)
    sns.boxplot(data=iacs, x=crop_class_col, y=farm_size_col, showfliers=False, ax=axs[1])
    axs[1].set_xlabel("Crop class")
    axs[1].set_ylabel("Farm size")
    axs[1].set_title("a) Farm sizes", loc="left")
    axs[1].grid(visible=True, which="major", axis="y", zorder=0)
    axs[1].set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(fr"figures\boxplots_field_and_farm_sizes_per_crop_class.png")

    crop_names = list(iacs[crop_class_col].unique())
    iacs["farm_size_r"] = iacs["farm_size"] - iacs["field_size"]
    iacs["log_farm_size"] = np.log(iacs[farm_size_col])
    iacs["log_field_size"] = np.log(iacs[field_size_col])
    iacs["log_farm_size_r"] = np.log(iacs["farm_size_r"])

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=plotting_lib.cm2inch((32, 32)))
    for cn, ax in zip(crop_names, axs.ravel()):
        sub = iacs.loc[iacs[crop_class_col] == cn].copy()
        sns.scatterplot(data=sub, x="log_field_size", y="log_farm_size", s=3, ax=ax)
        ax.set_title(cn.upper())
        ax.set_xlabel("Log field size")
        ax.set_ylabel("Log farm size")
    fig.tight_layout()
    plt.savefig(fr"figures\scatter_log_field_vs_log_farm_sizes_per_crop_class.png")

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=plotting_lib.cm2inch((32, 32)))
    for cn, ax in zip(crop_names, axs.ravel()):
        sub = iacs.loc[iacs[crop_class_col] == cn].copy()
        sns.scatterplot(data=sub, x="log_field_size", y="log_farm_size_r", s=3, ax=ax)
        ax.set_title(cn.upper())
        ax.set_xlabel("Log field size")
        ax.set_ylabel("Log farm size r")
    fig.tight_layout()
    plt.savefig(fr"figures\scatter_log_field_vs_log_rest_farm_sizes_per_crop_class.png")

    print("done!")


def farm_field_regression_in_hexagons(hex_shp, iacs, hexa_id_col, farm_id_col, field_id_col=None, field_size_col=None,
                                      farm_size_col=None, log_x=False, log_y=False, reduce_farm_size=True):
    print("Farm vs. field size regression.")

    ## Calculate centroids of fields to assign each field to a hexagon based on the centroid location
    print("\tAssign fields to hexagons.")
    iacs["centroids"] = iacs["geometry"].centroid
    if field_id_col:
        iacs.rename(columns={field_id_col: "field_id"}, inplace=True)
    else:
        iacs["field_id"] = iacs.index + 1
    centroids = iacs[["field_id", "centroids"]].copy()
    centroids.rename(columns={"centroids": "geometry"}, inplace=True)
    ## Create a centroids layer and add the hex id to the centroids
    centroids = gpd.GeoDataFrame(centroids, crs=25832)
    centroids = gpd.sjoin(centroids, hex_shp, how="left")

    ## Add hexagon id to iacs data by joining the centroids values with the
    iacs = pd.merge(iacs, centroids[["field_id", hexa_id_col]], how="left", on="field_id")

    ## Calculate field and farm sizes if not already provided
    print("\tDetermine field and farm sizes.")
    if field_size_col:
        iacs.rename(columns={field_size_col: "field_size"}, inplace=True)
    else:
        iacs["field_size"] = iacs["geometry"].area / 10000

    if farm_size_col:
        iacs.rename(columns={farm_size_col: "farm_size"}, inplace=True)
    else:
        farm_sizes = iacs.groupby(farm_id_col).agg(
            farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
        ).reset_index()
        iacs = pd.merge(iacs, farm_sizes, on=farm_id_col, how="left")

    # test for number of farm sizes per farm id
    t = iacs.groupby("farm_id").agg(
        n_farm_sizes=pd.NamedAgg("farm_size", "nunique")
    ).reset_index()
    t2 = t.loc[t["n_farm_sizes"] > 1].copy()
    if not t2.empty:
        warnings.warn("There are cases of farm_id's that have multiple farm sizes!")

    if reduce_farm_size:
        iacs["farm_size"] = iacs["farm_size"] - iacs["field_size"]

    ## Calculate regression explaining farm sizes with field sizes
    print("\tCalculate the regression.")
    def calc_regression(group, log_x=log_x, log_y=log_y):
        group = pd.DataFrame(group)
        group = group.loc[group["farm_size"] > 0].copy()

        if group.empty:
            return (np.nan, np.nan, np.nan, np.nan)

        if log_x:
            group['field_size'] = np.log(group['field_size'])
        if log_y:
            group['farm_size'] = np.log(group['farm_size'])

        # group.to_csv(r"C:\Users\IAMO\OneDrive - IAMO\2022_12 - Field vs farm sizes\data\tables\test2.csv")

        x = np.array(group["field_size"])
        y = np.array(group["farm_size"]).T

        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        estimation = model.fit()

        if len(estimation.params) > 1:
            slope = estimation.params[1]
            intercept = estimation.params[0]
            rsquared = estimation.rsquared
        else:
            slope = 0
            intercept = 1
            rsquared = 0

        return (intercept, slope, len(x), rsquared)

    ## Check whats wrong with some cases
    # group = iacs.loc[iacs[hexa_id_col] == 67].copy()
    # t = iacs.loc[iacs["field_id"] == "LS_8844", "farm_id"].iloc[0]
    # s = iacs.loc[iacs["farm_id"] == t].copy() #--> turns out we have to filter out farms with one field

    ## turn ha into sqm
    iacs["farm_size"] = iacs["farm_size"] * 10000
    iacs["field_size"] = iacs["field_size"] * 10000

    model_results = iacs.groupby(hexa_id_col).apply(calc_regression).reset_index()
    model_results.columns = [hexa_id_col, "model_results"]
    model_results["intercept"] = model_results["model_results"].apply(lambda x: x[0])
    model_results["slope"] = model_results["model_results"].apply(lambda x: x[1])
    model_results["num_fields"] = model_results["model_results"].apply(lambda x: x[2])
    model_results["rsquared"] = model_results["model_results"].apply(lambda x: x[3])
    model_results.drop(columns=["model_results"], inplace=True)

    mean_sizes = iacs.groupby(hexa_id_col).agg(
        avgfarm_s=pd.NamedAgg(column="farm_size", aggfunc="mean"),
        avgfield_s=pd.NamedAgg(column="field_size", aggfunc="mean"),
        num_farms=pd.NamedAgg(column=farm_id_col, aggfunc=list)
    ).reset_index()
    mean_sizes["num_farms"] = mean_sizes["num_farms"].apply(lambda x: len(set(x)))

    ## Add hexagon geometries to model results
    model_results = pd.merge(model_results, mean_sizes, how="left", on=hexa_id_col)
    model_results = pd.merge(model_results, hex_shp, how="left", on=hexa_id_col)
    model_results = gpd.GeoDataFrame(model_results)

    return model_results


def compare_nfk_and_sqr(nfk_pth, sqr_pth, out_pth):

    nfk_ras = gdal.Open(nfk_pth)
    nfk_ndv = nfk_ras.GetRasterBand(1).GetNoDataValue()
    nfk = nfk_ras.ReadAsArray()

    sqr_ras = gdal.Open(sqr_pth)
    sqr_ndv = sqr_ras.GetRasterBand(1).GetNoDataValue()
    sqr = sqr_ras.ReadAsArray()

    ndv_mask = nfk.copy()
    ndv_mask[ndv_mask != nfk_ndv] = 1
    ndv_mask[ndv_mask == nfk_ndv] = 0
    ndv_mask[sqr == sqr_ndv] = 0

    nfk[ndv_mask == 0] = np.nan
    sqr[ndv_mask == 0] = np.nan

    df = pd.DataFrame({"nfk": nfk.flatten(), "sqr": sqr.flatten()})
    df.dropna(inplace=True)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="nfk", y="sqr", s=1, edgecolor="none", ax=ax)
    ax.annotate(f"Corr. :{round(df['nfk'].corr(df['sqr']), 2)}", (300, 20))
    fig.tight_layout()
    plt.savefig(out_pth)
    print("done!")

## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    ## Only locally available
    input_dict = {
        "BB": {
            "iacs_pth": r"data\vector\IACS\IACS_BB_2018.shp",
            "farm_id_col": "BNR_ZD",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "K_ART_K"
        },
        "SA": {
            "iacs_pth": r"data\vector\IACS\IACS_SA_2018.shp",
            "farm_id_col": "btnr",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "NU_BEZ"
        },
        "TH": {
            "iacs_pth": r"data\vector\IACS\IACS_TH_2018.shp",
            "farm_id_col": "BDF_PI",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "BDF_KLA_"
        },
        "LS": {
            "iacs_pth": r"data\vector\IACS\IACS_LS_2018.shp",
            "farm_id_col": "REGISTRIER",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "K_ART_UNI"
        },
        "BV": {
            "iacs_pth": r"data\vector\IACS\IACS_BV_2018.shp",
            "farm_id_col": "bnrhash",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "beschreibu"
        }
    }

    key = "ALL"

    ## Without grasslands
    # pth = fr"data\vector\IACS\IACS_{key}_2018_new.shp"
    # iacs = prepare_large_iacs_shp(
    #     input_dict=input_dict,
    #     out_pth=pth
    # )
    # iacs = gpd.read_file(pth)
    # explore_farm_distribution_across_states(
    #     iacs=iacs,
    #     shp_out_pth=r"data\vector\IACS\IACS_2018_farms_in_multiple_states_no_grassland.shp",
    #     csv_out_pth=fr"data\tables\farm_distribution_across_states_no_grassland.csv",
    #     plt_out_pth=r"figures\share_of_farm_outside_main_state_no_grassland.png")

    ## With grasslands
    # pth = fr"data\vector\IACS\IACS_{key}_2018_with_grassland.shp"
    # iacs_grass = prepare_large_iacs_shp(
    #     input_dict=input_dict,
    #     out_pth=pth,
    #     grassland=True
    # )
    # iacs = gpd.read_file(pth)

    ## Reclassify grassland class
    # get_unqiue_crop_names_and_crop_classes_relations(
        # iacs=iacs,
        # crop_name_col="crop_name",
        # crop_class_col="ID_KTYP",
        # out_pth=rf"data\tables\unique_crop_name_class_relations.csv")

    # classification_df = pd.read_excel(r"data\tables\crop_names_to_new_classes.xlsx")
    iacs = new_crop_names_classification(
        iacs=iacs,
        crop_name_col="crop_name",
        crop_class_col="ID_KTYP",
        classification_df=classification_df,
        new_crop_class_col="new_ID_KTYP",
        out_pth=fr"data\vector\IACS\IACS_{key}_2018_with_grassland_recl.shp",
        validity_check_pth=fr"data\tables\IACS_{key}_2018_with_grassland_recl_validitiy_check.csv")

    iacs = gpd.read_file(fr"data\vector\IACS\IACS_{key}_2018_with_grassland_recl.shp")
    # explore_grasslands(
    #     iacs=iacs,
    #     farm_id_col="farm_id",
    #     field_id_col="field_id",
    #     field_size_col="field_size",
    #     farm_size_col="farm_size",
    #     out_pth=r"data\tables\grassland_statistics.xlsx"
    # )
    #
    # explore_farm_distribution_across_states(
    #     iacs=iacs,
    #     shp_out_pth=r"data\vector\IACS\IACS_2018_farms_in_multiple_states.shp",
    #     csv_out_pth=fr"data\tables\farm_distribution_across_states.csv",
    #     plt_out_pth=r"figures\share_of_farm_outside_main_state_png"
    # )
    #
    # explore_field_sizes(
    #     iacs=iacs,
    #     farm_id_col="farm_id",
    #     field_id_col="field_id",
    #     field_size_col="field_size",
    #     farm_size_col="farm_size",
    #     crop_class_col="ID_KTYP"
    # )

    ## Get some basic statistics
    # iacs["fstate"] = iacs["field_id"].apply(lambda x: x.split('_')[0])
    # df_stat = iacs.groupby("fstate").agg(
    #     area=pd.NamedAgg(column="field_size", aggfunc="sum"),
    #     num_fields=pd.NamedAgg(column="field_id", aggfunc="count"),
    #     num_farms=pd.NamedAgg(column="farm_id", aggfunc=list)
    # ).reset_index()
    # df_stat["num_farms"] = df_stat["num_farms"].apply(lambda x: len(set(x)))
    # df_stat["mean_field_size"] = df_stat["area"] / df_stat["num_fields"]
    # df_stat["mean_farm_size"] = df_stat["area"] / df_stat["num_farms"]
    #
    # df_stat.to_csv(r"data\tables\general_stats.csv", index=False)
    # print("Done calculating general statistics.")

    # compare_nfk_and_sqr(
    #     nfk_pth=r"Q:\FORLand\Field_farm_size_relationship\data\raster\NFKWe1000_250_3035.tif",
    #     sqr_pth=r"Q:\FORLand\Field_farm_size_relationship\data\raster\sqr1000_250_v10_3035.tif",
    #     out_pth=r"Q:\FORLand\Field_farm_size_relationship\figures\nfk_vs_sqr.png"
    # )

    ## Predict farm sizes based on field sizes for different hexagons layed over the study region
    for hexasize in [5, 1]: # 30, 15,
        hexagon_pth = fr"data\vector\grid\hexagon_grid_germany_{hexasize}km.shp"
        hex_shp = gpd.read_file(hexagon_pth)
        hex_shp.rename(columns={"id": "hexa_id"}, inplace=True)
        model_results_shp = farm_field_regression_in_hexagons(
            hex_shp=hex_shp,
            iacs=iacs,
            hexa_id_col="hexa_id",
            farm_id_col="farm_id",
            field_id_col="field_id",
            field_size_col="field_size",
            farm_size_col="farm_size",
            log_x=True,
            log_y=True,
            reduce_farm_size=True
        )
        pth = fr"data\vector\grid\hexagon_grid_{key}_{hexasize}km_with_values.shp"
        model_results_shp.to_file(pth)

    ## Plot the modelling results as maps
    for hexasize in [30, 15, 5, 1]:
        pth = fr"data\vector\grid\hexagon_grid_{key}_{hexasize}km_with_values.shp"
        model_results_shp = gpd.read_file(pth)
        model_results_shp.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_results_shp.dropna(subset=["rsquared"], how="all", inplace=True)
        model_results_shp = model_results_shp[model_results_shp["num_farms"] > 1].copy()
        if hexasize == 1:
            dpi = 600
            figsize = (40, 24)
        else:
            dpi = 300
            figsize = (20, 12)

        plotting_lib.plot_maps_in_grid(
            shp=model_results_shp,
            out_pth=fr"figures\maps\{key}_model_results_logscaled_{hexasize}km.png",
            cols=["intercept", "slope", "rsquared"],
            nrow=1,
            ncol=3,
            figsize=figsize,
            dpi=dpi,
            shp2_pth=fr"data\vector\administrative\GER_bundeslaender.shp",
            titles=["intercept", "slope", "rsquared"],
            highlight_extremes=False
        )

        plotting_lib.plot_maps_in_grid(
            shp=model_results_shp,
            out_pth=fr"figures\maps\{key}_num_farms_fields+sizes_{hexasize}km.png",
            cols=["num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
            nrow=1,
            ncol=4,
            figsize=figsize,
            dpi=dpi,
            shp2_pth=fr"data\vector\administrative\GER_bundeslaender.shp",
            titles=["no. farms", "no. fields", "mean field size [ha]", "mean farm size [ha]"],
            highlight_extremes=False
        )

    ## Generate summary plots for the plotting results
    ## 1st prepare dataframe
    pth1 = fr"data\vector\grid\hexagon_grid_{key}_1km_with_values.shp"
    pth5 = fr"data\vector\grid\hexagon_grid_{key}_5km_with_values.shp"
    pth15 = fr"data\vector\grid\hexagon_grid_{key}_15km_with_values.shp"
    pth30 = fr"data\vector\grid\hexagon_grid_{key}_30km_with_values.shp"
    model_results_shp1 = gpd.read_file(pth1)
    model_results_shp5 = gpd.read_file(pth5)
    model_results_shp15 = gpd.read_file(pth15)
    model_results_shp30 = gpd.read_file(pth30)

    model_results_shp1["df"] = "1"
    model_results_shp5["df"] = "5"
    model_results_shp15["df"] = "15"
    model_results_shp30["df"] = "30"

    df = pd.concat([model_results_shp1, model_results_shp5, model_results_shp15, model_results_shp30], axis=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["rsquared"], how="all", inplace=True)
    df = df[df["num_farms"] > 1].copy()
    df["num_farms_bins"] = pd.cut(df["num_farms"], [0, 10, 25, 100, 250, 1000, 2500])

    ## Plot rsqured vs no. farms
    plotting_lib.scatterplot_two_columns(
        df=df,
        out_pth=fr"figures\num_farms_vs_rsquared.png",
        col1="num_farms",
        col2="rsquared",
        hue="df"
    )

    ## Plot model result distributions for each hexagon size
    plotting_lib.boxplots_in_grid(
        df=df,
        out_pth=fr"figures\boxplots_hexasize_rsquared2.png",
        value_cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
        category_col="df",
        nrow=7,
        ncol=1,
        figsize=(7, 21),
        x_labels=("hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]"),
        dpi=300)

    ## Plot model result distributions for number of farms
    plotting_lib.boxplots_in_grid(
        df=df,
        out_pth=fr"figures\boxplots_num_farms_bins_rsquared2.png",
        value_cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
        category_col="num_farms_bins",
        nrow=7,
        ncol=1,
        figsize=(7, 21),
        x_labels=("Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin"),
        dpi=300)


    # for var in ["slope", "intercept", "num_fields", "rsquared", "mean_field_size", "mean_farm_size"]:
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_{var}.png",
    #         col=var,
    #         shp2_pth=fr"data\vector\administrative\GER_bundeslaender.shp",
    #         highlight_extremes=False
    #     )


    # for key in input_dict:
    #     print(f"Current federeal state: {key}")
    #
    #     hexagon_pth = fr"data\vector\grid\hexagon_grid_germany_15km.shp"
    #     hex_shp = gpd.read_file(hexagon_pth)
    #     hex_shp.rename(columns={"id": "hexa_id"}, inplace=True)
    #
    #     # iacs_pth = r"vector\IACS\IACS_BB_2018.shp"
    #     iacs_pth = input_dict[key]["iacs_pth"]
    #     farm_id_col = input_dict[key]["farm_id_col"]
    #     field_size_col = input_dict[key]["field_size_col"]
    #     farm_size_col = input_dict[key]["farm_size_col"]
    #     field_id_col = input_dict[key]["field_id_col"]
    #
    #     iacs = gpd.read_file(iacs_pth)
    #
    #     model_results_shp = farm_field_regression_in_hexagons(
    #         hex_shp=hex_shp,
    #         iacs=iacs,
    #         hexa_id_col="hexa_id",
    #         farm_id_col=farm_id_col,
    #         field_id_col=field_id_col,
    #         field_size_col=field_size_col,
    #         farm_size_col=farm_size_col
    #     )
    #     pth = fr"data\vector\grid\hexagon_grid_{key}_15km_with_values.shp"
        # model_results_shp.to_file(pth)
    #
    #     model_results_shp = gpd.read_file(pth)
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_slope.png",
    #         col="slope",
    #         shp2_pth=fr"data\vector\administrative\{key}_bundesland.shp",
    #         highlight_extremes=False
    #     )
    #
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_intercept.png",
    #         col="intercept",
    #         shp2_pth=fr"data\vector\administrative\{key}_bundesland.shp",
    #         highlight_extremes=False
    #     )
    #
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_num_fields.png",
    #         col="num_fields",
    #         shp2_pth=fr"data\vector\administrative\{key}_bundesland.shp",
    #         highlight_extremes=False
    #     )

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
