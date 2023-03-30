## Authors: Clemens Jänicke
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


def prepare_large_iacs_shp(input_dict, out_pth):
    print("Join all IACS files together.")

    shp_lst = []
    for key in input_dict:
        print(f"\tProcessing IACS data of {key}")
        iacs_pth = input_dict[key]["iacs_pth"]
        farm_id_col = input_dict[key]["farm_id_col"]

        shp = gpd.read_file(iacs_pth)
        shp["field_id"] = shp.index
        shp["field_id"] = shp["field_id"].apply(lambda x: f"{key}_{x}")
        if "ID_KTYP" in list(shp.columns):
            shp["ID_KTYP"] = shp["ID_KTYP"].astype(int)
            shp = shp.loc[shp["ID_KTYP"].isin([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 60])].copy()
        shp = shp.loc[shp["geometry"] != None].copy()

        shp["field_size"] = shp["geometry"].area / 10000

        shp.rename(columns={farm_id_col: "farm_id"}, inplace=True)
        shp = shp[["field_id", "farm_id", "field_size", "ID_KTYP", "geometry"]].copy()

        shp_lst.append(shp)

    print("\tConcatenating all shapefiles and writing out.")
    out_shp = pd.concat(shp_lst, axis=0)
    del shp_lst

    ## Calculate farm sizes
    farm_sizes = out_shp.groupby("farm_id").agg(
        farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
    ).reset_index()
    out_shp = pd.merge(out_shp, farm_sizes, on="farm_id", how="left")
    out_shp = out_shp[["field_id", "farm_id", "field_size", "farm_size", "ID_KTYP", "geometry"]].copy()
    # t = out_shp.loc[out_shp["geometry"] != None].copy()

    out_shp.to_file(out_pth)
    print("done!")
    return out_shp


def explore_field_sizes(iacs, farm_id_col, field_id_col, field_size_col, farm_size_col):

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

    farms["range_field_sizes"] = farms["max_field_size"] - farms["min_field_size"]
    farms["range_field_ratio"] = farms["range_field_sizes"] / farms["farm_size"]
    farms["max_field_ratio"] = farms["max_field_size"] / farms["farm_size"]
    farms["min_field_ratio"] = farms["min_field_size"] / farms["farm_size"]
    farms["diff_ratios"] = farms["max_field_ratio"] - farms["min_field_ratio"]

    farms["log_max_field_ratio"] = np.log(farms["max_field_ratio"])
    farms["log_min_field_ratio"] = np.log(farms["min_field_ratio"])
    farms["log_farm_size"] = np.log(farms["farm_size"])

    farms["farm_size_class"] = pd.cut(farms["farm_size"], [0, 10, 25, 100, 250, 1000, 2500], labels=[10, 25, 100, 250, 1000, 2500])
    farms["farm_size_class"] = pd.qcut(x=farms["farm_size"], q=10, labels=[1,2,3,4,5,6,7,8,9,10])


    # fig, axs = plt.subplots(nrows=3, ncols=1)
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, hue="num_fields", ax=axs[0])
    # sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs[1])
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, ax=axs[2])
    # sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs[2])
    # fig.tight_layout()
    #
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="diff_ratios", s=3, ax=axs)
    # fig.tight_layout()
    #
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="range_field_ratio", s=3, ax=axs)
    # fig.tight_layout()
    #
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, ax=axs)
    # sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs)
    # axs.set_xlabel("Farm size [ha]")
    # axs.set_ylabel("ratio of max. (blue) and min. (orange) field sizes")
    # fig.tight_layout()
    # plt.savefig(fr"figures\scatterplot_max+min_field_ratios.png")

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
    plt.savefig(fr"figures\boxplot_min_max_field_sizes_per_farm_decile.png")

    print("done!")


def farm_field_regression_in_hexagons(hex_shp, iacs, hexa_id_col, farm_id_col, field_id_col=None, field_size_col=None,
                                      farm_size_col=None, log_x=False, log_y=False):
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

    iacs["farm_size_b"] = iacs[farm_size_col] - iacs[field_size_col]

    # test for number of farm sizes per farm id
    t = iacs.groupby("farm_id").agg(
        n_farm_sizes=pd.NamedAgg("farm_size", "nunique")
    ).reset_index()
    t2 = t.loc[t["n_farm_sizes"] > 1].copy()
    if not t2.empty:
        warnings.warn("There are cases of farm_id's that have multiple farm sizes!")

    ## Calculate regression explaining farm sizes with field sizes
    print("\tCalculate the regression.")
    def calc_regression(group, log_x=log_x, log_y=log_y):
        group = pd.DataFrame(group)
        if log_x:
            group['field_size'] = np.log(group['field_size'])
        if log_y:
            group['farm_size'] = np.log(group['farm_size'])

        # group.to_csv(r"C:\Users\IAMO\OneDrive - IAMO\2022_12 - Field vs farm sizes\data\tables\test2.csv")

        x = np.array(group["field_size"])
        y = np.array(group["farm_size_b"]).T

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
            "farm_size_col": None
        },
        "SA": {
            "iacs_pth": r"data\vector\IACS\IACS_SA_2018.shp",
            "farm_id_col": "btnr",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "TH": {
            "iacs_pth": r"data\vector\IACS\IACS_TH_2018.shp",
            "farm_id_col": "BDF_PI",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "LS": {
            "iacs_pth": r"data\vector\IACS\IACS_LS_2018.shp",
            "farm_id_col": "REGISTRIER",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "BV": {
            "iacs_pth": r"data\vector\IACS\IACS_BV_2018.shp",
            "farm_id_col": "bnrhash",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        }
    }

    key = "ALL"
    pth = fr"data\vector\IACS\IACS_{key}_2018_new.shp"

    # iacs = prepare_large_iacs_shp(
    #     input_dict=input_dict,
    #     out_pth=pth
    # )
    # iacs = gpd.read_file(pth)

    ## Explore field sizes
    # explore_field_sizes(
    #     iacs=iacs,
    #     farm_id_col="farm_id",
    #     field_id_col="field_id",
    #     field_size_col="field_size",
    #     farm_size_col="farm_size"
    # )

    # compare_nfk_and_sqr(
    #     nfk_pth=r"Q:\FORLand\Field_farm_size_relationship\data\raster\NFKWe1000_250_3035.tif",
    #     sqr_pth=r"Q:\FORLand\Field_farm_size_relationship\data\raster\sqr1000_250_v10_3035.tif",
    #     out_pth=r"Q:\FORLand\Field_farm_size_relationship\figures\nfk_vs_sqr.png"
    # )

    # ## Get some basic statistics
    # iacs["fstate"] = iacs["field_id"].apply(lambda x: x.split('_')[0])
    # df_stat = iacs.groupby("fstate").agg(
    #     area=pd.NamedAgg(column="field_size",aggfunc="sum"),
    #     num_fields=pd.NamedAgg(column="field_id", aggfunc="count"),
    #     num_farms=pd.NamedAgg(column="farm_id", aggfunc=list)
    # ).reset_index()
    # df_stat["num_farms"] = df_stat["num_farms"].apply(lambda x: len(set(x)))
    # df_stat["mean_field_size"] = df_stat["area"] / df_stat["num_fields"]
    # df_stat["mean_farm_size"] = df_stat["area"] / df_stat["num_farms"]
    #
    # df_stat.to_csv(r"data\tables\general_stats.csv", index=False)
    print("Done calculating general statistics.")

    # for hexasize in [30, 15, 5, 1]:
    #     hexagon_pth = fr"data\vector\grid\hexagon_grid_germany_{hexasize}km.shp"
    #     hex_shp = gpd.read_file(hexagon_pth)
    #     hex_shp.rename(columns={"id": "hexa_id"}, inplace=True)
    #     model_results_shp = farm_field_regression_in_hexagons(
    #         hex_shp=hex_shp,
    #         iacs=iacs,
    #         hexa_id_col="hexa_id",
    #         farm_id_col="farm_id",
    #         field_id_col="field_id",
    #         field_size_col="field_size",
    #         farm_size_col="farm_size",
    #         log_x=True,
    #         log_y=True
    #     )
    #     pth = fr"data\vector\grid\hexagon_grid_{key}_{hexasize}km_with_values.shp"
    #     model_results_shp.to_file(pth)

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

    # pth1 = fr"data\vector\grid\hexagon_grid_{key}_1km_with_values.shp"
    # pth5 = fr"data\vector\grid\hexagon_grid_{key}_5km_with_values.shp"
    # pth15 = fr"data\vector\grid\hexagon_grid_{key}_15km_with_values.shp"
    # pth30 = fr"data\vector\grid\hexagon_grid_{key}_30km_with_values.shp"
    # model_results_shp1 = gpd.read_file(pth1)
    # model_results_shp5 = gpd.read_file(pth5)
    # model_results_shp15 = gpd.read_file(pth15)
    # model_results_shp30 = gpd.read_file(pth30)
    #
    # model_results_shp1["df"] = "1"
    # model_results_shp5["df"] = "5"
    # model_results_shp15["df"] = "15"
    # model_results_shp30["df"] = "30"
    #
    # df = pd.concat([model_results_shp1, model_results_shp5, model_results_shp15, model_results_shp30], axis=0)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.dropna(subset=["rsquared"], how="all", inplace=True)
    # df = df[df["num_farms"] > 1].copy()
    # df["num_farms_bins"] = pd.cut(df["num_farms"], [0, 10, 25, 100, 250, 1000, 2500])
    #
    # fig, ax = plt.subplots(1, 1, figsize=plotting_lib.cm2inch(15, 10))
    # sns.jointplot(data=df,  x="num_farms", y="rsquared", kind="kde")
    # fig.tight_layout()
    # plt.savefig(fr"figures\num_farms_vs_rsquared.png", dpi=300)
    # plt.close()
    #
    # plotting_lib.scatterplot_two_columns(
    #     df=df,
    #     out_pth=fr"figures\num_farms_vs_rsquared.png",
    #     col1="num_farms",
    #     col2="rsquared",
    #     hue="df"
    # )
    #
    # plotting_lib.boxplots_in_grid(
    #     df=df,
    #     out_pth=fr"figures\boxplots_hexasize_rsquared2.png",
    #     value_cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
    #     category_col="df",
    #     nrow=7,
    #     ncol=1,
    #     figsize=(7, 21),
    #     x_labels=("hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]"),
    #     dpi=300)
    #
    # plotting_lib.boxplots_in_grid(
    #     df=df,
    #     out_pth=fr"figures\boxplots_num_farms_bins_rsquared2.png",
    #     value_cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
    #     category_col="num_farms_bins",
    #     nrow=7,
    #     ncol=1,
    #     figsize=(7, 21),
    #     x_labels=("Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin"),
    #     dpi=300)


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
