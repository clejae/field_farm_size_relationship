import time
import os
import geopandas as gpd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
import matplotlib as mpl

import plotting_lib

WD = r"Q:\FORLand\Field_farm_size_relationship"
os.chdir(WD)
def accuracy_assessment(df, pred_cols, ref_cols, out_pth):

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=1, figsize=plotting_lib.cm2inch(30, 15 * len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]
        df_curr = df.dropna(subset=[pred_col]).copy()
        df_curr[ref_col] = round(df_curr[ref_col], 1)
        df_curr = df_curr.loc[df_curr[ref_col] > 0].copy()
        t = df_curr.loc[df_curr[ref_col] > 6000].copy()
        print(len(t))

        if len(pred_cols) == 1:
            ax = axs
        else:
            ax = axs[i]

        predictions = df_curr[pred_col]
        reference = df_curr[ref_col]

        n = len(df_curr)
        errors = abs(predictions - reference)
        me = round(np.mean(errors), 2)
        mse = mean_squared_error(reference, predictions)
        rmse = math.sqrt(mse)
        ape = (errors / reference) * 100
        mape = np.mean(ape)

        sns.scatterplot(y=predictions, x=reference, s=1, ax=ax)

        # Calculate the limits for the plot
        min_val = min(min(reference), min(predictions))
        max_val = max(max(reference), max(predictions))

        # Plot the 1-to-1 line
        ax.plot([min_val, max_val], [min_val, max_val], color='grey', label='1-to-1 Line')

        ax.set_ylabel("Farm size prediction [ha]")
        ax.set_xlabel("Farm size reference [ha]")

        # text = f"{n:,} fields\nMean bias: {round(a, 1)}\nPrecision: {round(p, 1)}\nRMSE: {round(u, 1)}\nMAPE: {round(mape, 1)}"
        text = f"{n:,} fields\nME: {round(me, 1)}\nRMSE: {round(rmse, 1)}\nMAPE: {round(mape, 1)}"
        ax.annotate(text=text, xy=(.1, .8), xycoords="axes fraction")
        # ax.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()
    plt.savefig(out_pth)
    plt.close()

def plot_apu(preds, pred_cols, ref_cols, apu_plot_pth, subtitles=None, qcut_col=None, cat_col=None):
    """
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JD009662
    The accuracy (A) statistically represents the mean bias of the estimates, μe, versus the truth data, μt,
    and is computed as:

    ref <- X
    est <- Y
    res <- est-ref
    npix <- length(res)
    nbp <- npix
    sumres <- sum(res)
    A <- sumres/nbp

    The precision, P, is representative of the repeatability of the estimate and is computed as
    the standard deviation of the estimates around the true values corrected for the mean bias (accuracy)

    sumresA_sq <- sum((res-A)*(res-A))
    P <- sqrt(sumresA_sq/(nbp-1))

    The uncertainty, U, represents the actual statistical deviation of the estimate from the truth
    including the mean bias and is computed as:

    sumres_sq <- sum(res*res)
    U <- sqrt(sumres_sq/nbp)

    :return:
    """

    # iacs[["field_id", "farm_size", "log_farm_size", "rfr_pred", "rfr_pred_log", "errors", "geometry"]]

    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri']

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=1, figsize=plotting_lib.cm2inch(16, 7 * len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]
        if subtitles:
            title = subtitles[i]
        else:
            title = pred_col
        res_dict = {"bin": [], "n": [], "a": [], "p": [], "u": [], "mape": []}
        preds_curr = preds.dropna(subset=[pred_col]).copy()
        preds_curr[ref_col] = round(preds_curr[ref_col], 1)
        preds_curr = preds_curr.loc[preds_curr[ref_col] > 0].copy()

        ## Discretize farm sizes into ventiles
        if qcut_col:
            preds_curr["bins"] = pd.qcut(preds_curr[qcut_col], q=10)
        elif cat_col:
            preds_curr["bins"] = preds_curr[cat_col]
        else:
            bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                                 (500, 1000), (1000, 7000)])
            preds_curr["bins"] = pd.cut(preds_curr[ref_col], bins=bins)


        for bin in preds_curr["bins"].unique():
            df_sub = preds_curr.loc[preds_curr["bins"] == bin].copy()
            n = len(df_sub)
            res = df_sub[pred_col] - df_sub[ref_col]
            a = np.sum(res) / n
            p = np.sqrt(np.sum((res - a) * (res - a)) / (n - 1))
            u = np.sqrt(np.sum(res * res) / n)

            errors = abs(df_sub[pred_col] - df_sub[ref_col])
            ape = (errors / df_sub[ref_col]) * 100
            mape = np.mean(ape)
            if mape > 5000:
                df_sub["residues"] = df_sub[pred_col] - df_sub[ref_col]
                df_sub["errors"] = abs(df_sub[pred_col] - df_sub[ref_col])
                df_sub["mape"] = (df_sub["errors"] / df_sub[ref_col]) * 100
                df_sub.sort_values(by="mape", ascending=False, inplace=True)
                ape.max()

            res_dict["bin"].append(bin)
            res_dict["n"].append(n)
            res_dict["a"].append(a)
            res_dict["p"].append(p)
            res_dict["u"].append(u)
            res_dict["mape"].append(mape)

        apu_df = pd.DataFrame.from_dict(res_dict)
        apu_df.sort_values(by="bin", inplace=True)
        apu_df.index = range(len(apu_df))

        # if cat_col_sorter:
        #     apu_df = apu_df.set_index('bin')
        #     apu_df.loc[cat_col_sorter]
        #     apu_df["bin"] = cat_col_sorter
            # apu_df.sort_values(by="bin", key=lambda column: column.map(lambda e: cat_col_sorter.index(e)), inplace=True)

        n = len(preds_curr)
        # n_farms = (preds_curr["farm_id"].unique())
        res = preds_curr[pred_col] - preds_curr[ref_col]
        a = np.sum(res) / n
        p = np.sqrt(np.sum((res - a) * (res - a)) / (n - 1))
        u = np.sqrt(np.sum(res * res) / n)

        errors = abs(preds_curr[pred_col] - preds_curr[ref_col])
        ape = (errors / preds_curr[ref_col]) * 100
        mape = np.mean(ape)

        if len(pred_cols) == 1:
            ax = axs
        else:
            ax = axs[i]

        ax.set_title(title, loc="left")

        ax2 = ax.twinx()
        sns.barplot(data=apu_df, x='bin', y='n', alpha=0.5, ax=ax2, facecolor="white", edgecolor="black", zorder=0)
        ax2.set_ylabel('Number of fields')

        # plt.setp(ax2.lines, zorder=0)
        # plt.setp(ax2.collections, zorder=0, label="")
        ax.axhline(y=0, color="black", linewidth=0.5)
        # if cat_col:
        #     sns.scatterplot(data=apu_df, x="bin", y="a", marker='o', ax=ax, color="blue", zorder=1)
        #     sns.scatterplot(data=apu_df, x="bin", y="p", marker='o', ax=ax, color="orange", zorder=1)
        #     sns.scatterplot(data=apu_df, x="bin", y="u", marker='o', ax=ax, color="green", zorder=1)
        #     sns.scatterplot(data=apu_df, x="bin", y="mape", marker='o', ax=ax, color="grey")
        # else:
        sns.lineplot(data=apu_df["a"], marker='o', sort=False, ax=ax, color="blue", zorder=1)
        sns.lineplot(data=apu_df["p"], marker='o', sort=False, ax=ax, color="orange", zorder=1)
        sns.lineplot(data=apu_df["u"], marker='o', sort=False, ax=ax, color="green", zorder=1)
        sns.lineplot(data=apu_df["mape"], marker='o', sort=False, ax=ax, color="grey")

        ax.set_ylabel('ME/RMSE/MAPE/P')

        text = f"{n:,} fields\nME: {round(a, 1)}\nRMSE: {round(u, 1)}\nMAPE: {round(mape, 1)}\nP: {round(p, 1)}"
        ax.annotate(text=text, xy=(.15, .65), xycoords="axes fraction", zorder=1)

        ax.patch.set_visible(False)

        ax.set_zorder(ax.get_zorder() + 100)
        ax.set_zorder(ax2.get_zorder() + 100)

        # plt.setp(ax.lines, zorder=100)
        # plt.setp(ax.collections, zorder=100, label="")
        # ax.yaxis.set_major_formatter(StrMethodFormatter('{int(x):,}'))
        # ax2.yaxis.set_major_formatter(StrMethodFormatter('{int(x):,}'))
        formater = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
        ax.yaxis.set_major_formatter(formater)
        ax2.yaxis.set_major_formatter(formater)

        if (len(pred_cols) != 1) & (i < len(pred_cols)-1):
            ax.set_xticklabels([])

    ## Add this only to last plot
    ax.set_xlabel('Farm sizes [ha]')
    ax.tick_params(axis='x', labelrotation=45)

    ## Add this only to first plot
    if len(pred_cols) > 1:
        ax = axs[0]
    legend_elements = [Patch(facecolor="blue", edgecolor=None, label="ME"),
                       Patch(facecolor="green", edgecolor=None, label="RMSE"),
                       Patch(facecolor="grey", edgecolor=None, label="MAPE"),
                       Patch(facecolor="orange", edgecolor=None, label="P")]
    ax.legend(handles=legend_elements, bbox_to_anchor=(.65, .6), ncol=1, title=None)
    # bbox = ax.get_position()
    #
    # # Place the legend below this subplot using the custom legend elements
    # fig.legend(handles=legend_elements, loc='upper center',
    #            bbox_to_anchor=(bbox.x0 + bbox.width / 2, bbox.y0 - 0.15), ncol=2)

    fig.tight_layout()
    plt.savefig(apu_plot_pth)
    plt.close()

    print("Plotting done!")


def assign_hexa_ids_to_field_ids(hex_shp_pth, iacs_pth, out_pth):

    hex_shp = gpd.read_file(hex_shp_pth)
    iacs = gpd.read_file(iacs_pth)


    iacs["centroids"] = iacs["geometry"].centroid
    centroids = iacs[["field_id", "centroids"]].copy()
    centroids.rename(columns={"centroids": "geometry"}, inplace=True)
    ## Create a centroids layer and add the hex id to the centroids
    centroids = gpd.GeoDataFrame(centroids, crs=3035)
    centroids = centroids.to_crs(25832)
    centroids = gpd.sjoin(centroids, hex_shp, how="left")
    centroids[["field_id", "hexa_id"]].to_csv(out_pth, index=False)

    print("done.")


def plot_aggregated_farm_sizes_hexagon(preds, hex_shp, f_to_h, pred_cols, ref_cols, out_pth, gdf_2, gdf_3):

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=4, figsize=plotting_lib.cm2inch(30, 10 * len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]

        df_curr = preds.dropna(subset=[pred_col]).copy()
        df_curr["curr_error"] = df_curr[ref_col] - df_curr[pred_col]
        df_curr["federal_state"] = df_curr["field_id"].apply(lambda x: x.split("_")[0])

        df_comb = pd.merge(df_curr, f_to_h, how="left", on="field_id")
        df_plt = df_comb.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
            agg_std=pd.NamedAgg(pred_col, "std")
        ).reset_index()

        ## All categories as APU plot
        labels = ["(0, 5]", "(5, 10]", "(10, 20]", "(20, 50]", "(50, 100]", "(100, 200]", "(200, 500]", "(500, 1000]",
                  "(1000, 7000]"]
        bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                             (500, 1000), (1000, 7000)])

        ## All categories that actually appear in data
        labels = ["(10, 20]", "(20, 50]", "(50, 100]", "(100, 200]", "(200, 500]", "(500, 1000]",
                  "(1000, 7000]"]
        bins = pd.IntervalIndex.from_tuples([(10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                             (500, 1000), (1000, 7000)])

        colors = plt.get_cmap('viridis').colors
        step = len(colors) // len(labels)
        color_dict = {curr_bin: colors[i * step] for i, curr_bin in enumerate(labels)}

        df_plt["bins_ref"] = pd.cut(df_plt["agg_ref"], bins=bins, labels=labels)
        df_plt["bins_ref"] = df_plt["bins_ref"].astype(str)
        df_plt["bins_pred"] = pd.cut(df_plt["agg_pred"], bins=bins, labels=labels)
        df_plt["bins_pred"] = df_plt["bins_pred"].astype(str)

        gdf = pd.merge(hex_shp, df_plt, how="left", on="hexa_id")
        gdf.dropna(inplace=True)

        # Get the range of values for colormap normalization
        vmin = min(gdf["agg_ref"].min(), gdf["agg_pred"].min())
        vmax = max(gdf["agg_ref"].max(), gdf["agg_pred"].max())

        vmin = (vmax - vmin) / 100 * 2
        vmax = (vmax - vmin) / 100 * 98

        # Plot the reference and prediction maps
        axs[i, 0].set_title("Reference")
        gdf.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin, vmax=vmax, legend=True,
        legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        # gdf.plot(column="bins_ref", ax=axs[i, 0], colors)
        # gdf["color"] = gdf["bins_ref"].map(color_dict)
        # gdf.plot(color=gdf["color"], ax=axs[i, 0], legend=False, edgecolor='none')
        axs[i, 0].axis('off')

        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 0], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 0], lw=0.7, zorder=2)

        axs[i, 1].set_title("Prediction")
        gdf.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin, vmax=vmax, legend=True,
        legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        # gdf.plot(column="bins_pred", ax=axs[i, 1])
        # gdf["color"] = gdf["bins_pred"].map(color_dict)
        # gdf.plot(color=gdf["color"], ax=axs[i, 1], legend=False, edgecolor='none')
        axs[i, 1].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 1], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 1], lw=0.7, zorder=2)

        axs[i, 2].set_title("Error")
        gdf.plot(column="agg_err", ax=axs[i, 2], legend=True,
        legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        axs[i, 2].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 2], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 2], lw=0.7, zorder=2)

        predictions = gdf["agg_pred"]
        reference = gdf["agg_ref"]

        errors = abs(predictions - reference)
        mae = round(np.mean(errors), 2)
        mse = mean_squared_error(reference, predictions)
        rmse = math.sqrt(mse)
        mape = (errors / reference) * 100
        accuracy = 100 - np.mean(mape)

        sns.scatterplot(x=predictions, y=reference, s=1, ax=axs[i, 3])

        # Calculate the limits for the plot
        min_val = min(min(reference), min(predictions))
        max_val = max(max(reference), max(predictions))

        # Plot the 1-to-1 line
        axs[i, 3].plot([min_val, max_val], [min_val, max_val], color='grey', label='1-to-1 Line')

        axs[i, 3].set_ylabel("Farm size prediction [ha]")
        axs[i, 3].set_xlabel("Farm size reference [ha]")

        text = f"MAE: {round(mae, 1)}\nMSE: {round(mse, 1)}\nRMSE: {round(rmse, 1)}, \nAccuracy: {round(accuracy, 1)}"
        axs[i, 3].annotate(text=text, xy=(.1, .8), xycoords="axes fraction")

    # Adjust layout to make space for a shared colorbar
    fig.tight_layout()
    # cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # This is the position for the colorbar
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm._A = []  # Necessary for scalar mappable
    # fig.colorbar(sm, cax=cax)

    plt.savefig(out_pth)
    plt.close()

    print("Plotting done!")

def plot_aggregated_farm_sizes_hexagon_divided(preds, hex_shp, f_to_h, pred_cols, ref_cols, out_pth, gdf_2, gdf_3):

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=4, figsize=plotting_lib.cm2inch(30, 10 * len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]

        df_curr = preds.dropna(subset=[pred_col]).copy()
        df_curr["curr_error"] = df_curr[ref_col] - df_curr[pred_col]
        df_curr["federal_state"] = df_curr["field_id"].apply(lambda x: x.split("_")[0])
        len(df_curr.loc[df_curr["federal_state"] == "BB"])
        df_east = df_curr.loc[df_curr["federal_state"].isin(["BB", "SA", "TH"])].copy()
        df_west = df_curr.loc[df_curr["federal_state"].isin(["BV", "LS"])].copy()
        df_comb = pd.merge(df_curr, f_to_h, how="left", on="field_id")
        df_comb_e = pd.merge(df_east, f_to_h, how="left", on="field_id")
        df_comb_w = pd.merge(df_west, f_to_h, how="left", on="field_id")
        df_plt_e = df_comb_e.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()

        df_plt_w = df_comb_w.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()

        df_plt = df_comb.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()

        ## All categories as APU plot
        labels = ["(0, 5]", "(5, 10]", "(10, 20]", "(20, 50]", "(50, 100]", "(100, 200]", "(200, 500]", "(500, 1000]",
                  "(1000, 7000]"]
        bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                             (500, 1000), (1000, 7000)])

        ## All categories that actually appear in data
        labels = ["(10, 20]", "(20, 50]", "(50, 100]", "(100, 200]", "(200, 500]", "(500, 1000]",
                  "(1000, 7000]"]
        bins = pd.IntervalIndex.from_tuples([(10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                             (500, 1000), (1000, 7000)])

        colors = plt.get_cmap('viridis').colors
        step = len(colors) // len(labels)
        color_dict = {curr_bin: colors[i * step] for i, curr_bin in enumerate(labels)}

        # df_plt["bins_ref"] = pd.cut(df_plt["agg_ref"], bins=bins, labels=labels)
        # df_plt["bins_ref"] = df_plt["bins_ref"].astype(str)
        # df_plt["bins_pred"] = pd.cut(df_plt["agg_pred"], bins=bins, labels=labels)
        # df_plt["bins_pred"] = df_plt["bins_pred"].astype(str)

        gdf_w = pd.merge(hex_shp, df_plt_w, how="left", on="hexa_id")
        gdf_w.dropna(inplace=True)

        gdf_e = pd.merge(hex_shp, df_plt_e, how="left", on="hexa_id")
        gdf_e.dropna(inplace=True)

        gdf = pd.merge(hex_shp, df_plt, how="left", on="hexa_id")
        gdf.dropna(inplace=True)

        # Get the range of values for colormap normalization
        vmin_e = min(gdf_e["agg_ref"].min(), gdf_e["agg_pred"].min())
        vmax_e = max(gdf_e["agg_ref"].max(), gdf_e["agg_pred"].max())

        vmin_e = gdf_e["agg_ref"].quantile(q=0.02)
        vmax_e = gdf_e["agg_ref"].quantile(q=0.98)

        vmin_w = min(gdf_w["agg_ref"].min(), gdf_w["agg_pred"].min())
        vmax_w = max(gdf_w["agg_ref"].max(), gdf_w["agg_pred"].max())

        vmin_w = gdf_w["agg_ref"].quantile(q=0.02)
        vmax_w = gdf_w["agg_ref"].quantile(q=0.98)

        # Plot the reference and prediction maps
        axs[i, 0].set_title("Reference")
        gdf_e.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin_e, vmax=vmax_e, legend=True, cmap="Blues",
                   legend_kwds={'label': "East", 'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin_w, vmax=vmax_w, legend=True, cmap="Reds",
                   legend_kwds={'label': "West", 'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 0.7), "pad": 0.01})
        # gdf["color"] = gdf["bins_ref"].map(color_dict)
        # gdf.plot(color=gdf["color"], ax=axs[i, 0], legend=False, edgecolor='none')
        axs[i, 0].axis('off')

        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 0], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 0], lw=0.7, zorder=2)

        axs[i, 1].set_title("Prediction")
        gdf_e.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin_e, vmax=vmax_e, legend=True, cmap="Blues",
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin_w, vmax=vmax_w, legend=True, cmap="Reds",
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.0), "pad": 0.01})
        # gdf["color"] = gdf["bins_pred"].map(color_dict)
        # gdf.plot(color=gdf["color"], ax=axs[i, 1], legend=False, edgecolor='none')
        axs[i, 1].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 1], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 1], lw=0.7, zorder=2)

        vmin_err = gdf["agg_err"].quantile(q=0.02)
        vmax_err = gdf["agg_err"].quantile(q=0.98)

        axs[i, 2].set_title("Error")
        vmin_e_err = gdf_e["agg_err"].quantile(q=0.02)
        vmax_e_err = gdf_e["agg_err"].quantile(q=0.98)
        vmin_w_err = gdf_w["agg_err"].quantile(q=0.02)
        vmax_w_err = gdf_w["agg_err"].quantile(q=0.98)
        gdf_e.plot(column="agg_err", ax=axs[i, 2], vmin=vmin_e_err, vmax=vmax_e_err, legend=True, cmap="Blues",
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_err", ax=axs[i, 2], vmin=vmin_w_err, vmax=vmax_w_err, legend=True, cmap="Reds",
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.0), "pad": 0.01})
        axs[i, 2].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 2], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 2], lw=0.7, zorder=2)

        predictions = gdf["agg_pred"]
        reference = gdf["agg_ref"]

        n = len(gdf)
        res = gdf["agg_pred"] - gdf["agg_ref"]
        a = np.sum(res) / n
        p = np.sqrt(np.sum((res - a) * (res - a)) / (n - 1))
        u = np.sqrt(np.sum(res * res) / n)

        errors = abs(gdf["agg_pred"] - gdf["agg_ref"])
        ape = (errors / gdf["agg_ref"]) * 100
        mape = np.mean(ape)

        sns.scatterplot(x=predictions, y=reference, s=1.5, ax=axs[i, 3])

        # Calculate the limits for the plot
        min_val = min(min(reference), min(predictions))
        max_val = max(max(reference), max(predictions))

        # Plot the 1-to-1 line
        axs[i, 3].plot([min_val, max_val], [min_val, max_val], color='grey', label='1-to-1 Line')

        axs[i, 3].set_ylabel("Farm size reference [ha]")
        axs[i, 3].set_xlabel("Farm size prediction [ha]")

        text = f"ME: {round(a, 1)}\nRMSE: {round(u, 1)}\nMAPE: {round(mape, 1)}\nP: {round(p, 1)}"
        # text = f"MAE: {round(mae, 1)}\nMSE: {round(mse, 1)}\nRMSE: {round(rmse, 1)}, \nAccuracy: {round(accuracy, 1)}"
        axs[i, 3].annotate(text=text, xy=(.1, .8), xycoords="axes fraction")

    # Adjust layout to make space for a shared colorbar
    fig.tight_layout()

    plt.savefig(out_pth)
    plt.close()

    ## Save legends
    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]

        df_curr = preds.dropna(subset=[pred_col]).copy()
        df_curr["curr_error"] = df_curr[ref_col] - df_curr[pred_col]
        df_curr["federal_state"] = df_curr["field_id"].apply(lambda x: x.split("_")[0])
        df_east = df_curr.loc[df_curr["federal_state"].isin(["BB", "SA", "TH"])].copy()
        df_west = df_curr.loc[df_curr["federal_state"].isin(["BV", "LS"])].copy()
        df_comb_e = pd.merge(df_east, f_to_h, how="left", on="field_id")
        df_comb_w = pd.merge(df_west, f_to_h, how="left", on="field_id")
        df_comb = pd.merge(df_curr, f_to_h, how="left", on="field_id")

        df_plt_e = df_comb_e.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()
        df_plt_w = df_comb_w.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()
        df_plt = df_comb.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()

        gdf_w = pd.merge(hex_shp, df_plt_w, how="left", on="hexa_id")
        gdf_w.dropna(inplace=True)
        gdf_e = pd.merge(hex_shp, df_plt_e, how="left", on="hexa_id")
        gdf_e.dropna(inplace=True)
        gdf = pd.merge(hex_shp, df_plt, how="left", on="hexa_id")
        gdf.dropna(inplace=True)

        vmin_e = gdf_e["agg_ref"].quantile(q=0.02)
        vmax_e = gdf_e["agg_ref"].quantile(q=0.98)
        vmin_w = gdf_w["agg_ref"].quantile(q=0.02)
        vmax_w = gdf_w["agg_ref"].quantile(q=0.98)
        vmin_err_e = gdf_e["agg_err"].quantile(q=0.02)
        vmax_err_e = gdf_e["agg_err"].quantile(q=0.98)
        vmin_err_w = gdf_w["agg_err"].quantile(q=0.02)
        vmax_err_w = gdf_w["agg_err"].quantile(q=0.98)

        ## Legend for reference and predictions
        fig_leg, (ax_leg_e, ax_leg_w) = plt.subplots(nrows=2, figsize=plotting_lib.cm2inch(7.5, 3))
        cmap_e = mpl.cm.Blues
        norm_e = mpl.colors.Normalize(vmin=vmin_e, vmax=vmax_e)
        cbar_e = mpl.colorbar.ColorbarBase(ax_leg_e, cmap=cmap_e,
                                           norm=norm_e,
                                           orientation='horizontal')
        # East
        cbar_e.ax.xaxis.set_ticks_position('top')

        cmap_w = mpl.cm.Reds
        norm_w = mpl.colors.Normalize(vmin=vmin_w, vmax=vmax_w)
        cbar_w = mpl.colorbar.ColorbarBase(ax_leg_w, cmap=cmap_w,
                                           norm=norm_w,
                                           orientation='horizontal')
        # West
        fig_leg.subplots_adjust(hspace=0.1)
        fig_leg.tight_layout()

        plt.savefig(out_pth[:-4] + f"_legend_{pred_col}.png")
        plt.close()

        ## Legend for error
        fig_leg, (ax_leg_e, ax_leg_w) = plt.subplots(nrows=2, figsize=plotting_lib.cm2inch(7.5, 3))
        cmap_e = mpl.cm.Blues
        norm_e = mpl.colors.Normalize(vmin=vmin_err_e, vmax=vmax_err_e)
        cbar_e = mpl.colorbar.ColorbarBase(ax_leg_e, cmap=cmap_e,
                                           norm=norm_e,
                                           orientation='horizontal')
        # East
        cbar_e.ax.xaxis.set_ticks_position('top')

        cmap_w = mpl.cm.Reds
        norm_w = mpl.colors.Normalize(vmin=vmin_err_w, vmax=vmax_err_w)
        cbar_w = mpl.colorbar.ColorbarBase(ax_leg_w, cmap=cmap_w,
                                           norm=norm_w,
                                           orientation='horizontal')
        # West
        fig_leg.subplots_adjust(hspace=0.1)
        fig_leg.tight_layout()

        plt.savefig(out_pth[:-4] + f"_legend_error_{pred_col}.png")
        plt.close()

    print("Plotting done!")

def plot_aggregated_farm_sizes_hexagon_divided_std(preds, hex_shp, f_to_h, pred_cols, ref_cols, out_pth, gdf_2, gdf_3):

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=4, figsize=plotting_lib.cm2inch(30, 10 * len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]

        df_curr = preds.dropna(subset=[pred_col]).copy()
        df_curr["curr_error"] = df_curr[ref_col] - df_curr[pred_col]
        df_curr["federal_state"] = df_curr["field_id"].apply(lambda x: x.split("_")[0])
        len(df_curr.loc[df_curr["federal_state"] == "BB"])
        df_east = df_curr.loc[df_curr["federal_state"].isin(["BB", "SA", "TH"])].copy()
        df_west = df_curr.loc[df_curr["federal_state"].isin(["BV", "LS"])].copy()
        df_comb = pd.merge(df_curr, f_to_h, how="left", on="field_id")
        df_comb_e = pd.merge(df_east, f_to_h, how="left", on="field_id")
        df_comb_w = pd.merge(df_west, f_to_h, how="left", on="field_id")
        df_plt_e = df_comb_e.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
            agg_std=pd.NamedAgg(pred_col, "std")
        ).reset_index()

        df_plt_w = df_comb_w.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
            agg_std=pd.NamedAgg(pred_col, "std")
        ).reset_index()

        df_plt = df_comb.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()

        gdf_w = pd.merge(hex_shp, df_plt_w, how="left", on="hexa_id")
        gdf_w.dropna(inplace=True)

        gdf_e = pd.merge(hex_shp, df_plt_e, how="left", on="hexa_id")
        gdf_e.dropna(inplace=True)

        gdf = pd.merge(hex_shp, df_plt, how="left", on="hexa_id")
        gdf.dropna(inplace=True)

        ## Ranges for colormap normalization
        vmin_e = gdf_e["agg_ref"].quantile(q=0.02)
        vmax_e = gdf_e["agg_ref"].quantile(q=0.98)
        vmin_w = gdf_w["agg_ref"].quantile(q=0.02)
        vmax_w = gdf_w["agg_ref"].quantile(q=0.98)

        # Plot the reference and prediction maps
        axs[i, 0].set_title("Reference")
        gdf_e.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin_e, vmax=vmax_e, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin_w, vmax=vmax_w, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.0), "pad": 0.01})
        axs[i, 0].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 0], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 0], lw=0.7, zorder=2)

        axs[i, 1].set_title("Prediction")
        gdf_e.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin_e, vmax=vmax_e, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin_w, vmax=vmax_w, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.0), "pad": 0.01})
        axs[i, 1].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 1], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 1], lw=0.7, zorder=2)

        axs[i, 2].set_title("Std. dev. prediction")
        vmin_e = gdf_e["agg_std"].quantile(q=0.05)
        vmax_e = gdf_e["agg_std"].quantile(q=0.98)
        vmin_w = gdf_w["agg_std"].quantile(q=0.05)
        vmax_w = gdf_w["agg_std"].quantile(q=0.98)
        gdf_e.plot(column="agg_std", ax=axs[i, 2], vmin=vmin_e, vmax=vmax_e, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_std", ax=axs[i, 2], vmin=vmin_w, vmax=vmax_w, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.0), "pad": 0.01})
        axs[i, 2].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 2], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 2], lw=0.7, zorder=2)

        axs[i, 3].set_title("Error")
        vmin_e_err = gdf_e["agg_err"].quantile(q=0.02)
        vmax_e_err = gdf_e["agg_err"].quantile(q=0.98)
        vmin_w_err = gdf_w["agg_err"].quantile(q=0.02)
        vmax_w_err = gdf_w["agg_err"].quantile(q=0.98)
        gdf_e.plot(column="agg_err", ax=axs[i, 3], vmin=vmin_e_err, vmax=vmax_e_err, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.5), "pad": 0.01})
        gdf_w.plot(column="agg_err", ax=axs[i, 3], vmin=vmin_w_err, vmax=vmax_w_err, legend=True,
                   legend_kwds={'orientation': "horizontal", "fraction": 0.04, "anchor": (0.1, 1.0), "pad": 0.01})
        axs[i, 3].axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=axs[i, 3], lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=axs[i, 3], lw=0.7, zorder=2)

    fig.tight_layout()

    plt.savefig(out_pth)
    plt.close()

    print("Plotting done!")

def plot_aggregated_farm_sizes_hexagon_final(preds, hex_shp, f_to_h, pred_cols, ref_cols, out_pth, gdf_2, gdf_3):

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=4, figsize=plotting_lib.cm2inch(30, 10 * len(pred_cols)))

    fig = plt.figure(figsize=plotting_lib.cm2inch(25, 12))
    # widths = [1, 1, 1, 1]
    heights = [5, 5, 1]
    gs = fig.add_gridspec(ncols=4, nrows=3, height_ratios=heights) #width_ratios=widths,

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, :])

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]

        df_curr = preds.dropna(subset=[pred_col]).copy()
        df_curr["curr_error"] = df_curr[ref_col] - df_curr[pred_col]
        df_comb = pd.merge(df_curr, f_to_h, how="left", on="field_id")
        df_plt = df_comb.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean"),
            agg_err=pd.NamedAgg("curr_error", "mean"),
        ).reset_index()

        ## All categories as APU plot
        labels = ["(0, 5]", "(5, 10]", "(10, 20]", "(20, 50]", "(50, 100]", "(100, 200]", "(200, 500]", "(500, 1000]",
                  "(1000, 7000]"]
        bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                             (500, 1000), (1000, 7000)])

        ## All categories that actually appear in data
        labels = ["(10, 20]", "(20, 50]", "(50, 100]", "(100, 200]", "(200, 500]", "(500, 1000]",
                  "(1000, 7000]"]
        bins = pd.IntervalIndex.from_tuples([(10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                             (500, 1000), (1000, 7000)])

        colors = plt.get_cmap('viridis').colors
        step = len(colors) // len(labels)
        color_dict = {curr_bin: colors[i * step] for i, curr_bin in enumerate(labels)}

        df_plt["bins_ref"] = pd.cut(df_plt["agg_ref"], bins=bins, labels=labels)
        df_plt["bins_ref"] = df_plt["bins_ref"].astype(str)
        df_plt["bins_pred"] = pd.cut(df_plt["agg_pred"], bins=bins, labels=labels)
        df_plt["bins_pred"] = df_plt["bins_pred"].astype(str)

        gdf = pd.merge(hex_shp, df_plt, how="left", on="hexa_id")
        gdf.dropna(inplace=True)

        # Get the range of values for colormap normalization
        vmin = min(gdf["agg_ref"].min(), gdf["agg_pred"].min())
        vmax = max(gdf["agg_ref"].max(), gdf["agg_pred"].max())

        # Plot the reference and prediction maps
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.set_title("Reference")
        # gdf.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin, vmax=vmax)
        # gdf.plot(column="bins_ref", ax=axs[i, 0], colors)
        gdf["color"] = gdf["bins_ref"].map(color_dict)
        gdf.plot(color=gdf["color"], ax=ax1, legend=False, edgecolor='none')
        ax1.axis('off')

        gdf_2.plot(edgecolor='black', facecolor="none", ax=ax1, lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=ax1, lw=0.7, zorder=2)

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.set_title("Prediction")
        # gdf.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin, vmax=vmax)
        # gdf.plot(column="bins_pred", ax=axs[i, 1])
        gdf["color"] = gdf["bins_pred"].map(color_dict)
        gdf.plot(color=gdf["color"], ax=ax2, legend=False, edgecolor='none')
        ax2.axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=ax2, lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=ax2, lw=0.7, zorder=2)

        ax3 = fig.add_subplot(gs[i, 2])
        ax3.set_title("Error")
        gdf.plot(column="agg_err", ax=ax3, legend=False)
        ax3.axis('off')
        gdf_2.plot(edgecolor='black', facecolor="none", ax=ax3, lw=0.1, zorder=2)
        gdf_3.plot(edgecolor='black', facecolor="none", ax=ax3, lw=0.7, zorder=2)

        ax4 = fig.add_subplot(gs[i, 3])
        predictions = gdf["agg_pred"]
        reference = gdf["agg_ref"]

        errors = abs(predictions - reference)
        mae = round(np.mean(errors), 2)
        mse = mean_squared_error(reference, predictions)
        rmse = math.sqrt(mse)
        mape = (errors / reference) * 100
        accuracy = 100 - np.mean(mape)

        sns.scatterplot(y=predictions, x=reference, s=1, ax=ax4)

        # Calculate the limits for the plot
        min_val = min(min(reference), min(predictions))
        max_val = max(max(reference), max(predictions))

        # Plot the 1-to-1 line
        ax4.plot([min_val, max_val], [min_val, max_val], color='grey', label='1-to-1 Line')

        ax4.set_ylabel("Farm size prediction [ha]")
        ax4.set_xlabel("Farm size reference [ha]")

        text = f"MAE: {round(mae, 1)}\nMSE: {round(mse, 1)}\nRMSE: {round(rmse, 1)}, \nAccuracy: {round(accuracy, 1)}"
        ax4.annotate(text=text, xy=(.1, .8), xycoords="axes fraction")

    # ax5 = fig.add_subplot(gs[i + 1, 1])
    # custom_patches = [Patch(facecolor=color_dict[v], label=v) for v in labels]
    # ax5.axis("off")
    # ax5.margins(0)
    # ax5.legend(handles=custom_patches, ncol=1, bbox_to_anchor=(1.07, 1))

    # ax6 = fig.add_subplot(gs[i + 1, 2])
    # cmap = mpl.cm.viridis
    # norm = mpl.colors.Normalize(vmin=gdf["agg_err"].min(), vmax=gdf["agg_err"].max())
    # cb1 = mpl.colorbar.ColorbarBase(ax6, cmap=cmap, norm=norm, orientation='horizontal')


    # Adjust layout to make space for a shared colorbar
    fig.tight_layout()
    # cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # This is the position for the colorbar
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm._A = []  # Necessary for scalar mappable
    # fig.colorbar(sm, cax=cax)

    plt.savefig(out_pth)
    plt.close()

    print("Plotting done!")

## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    ## Just once needed
    # assign_hexa_ids_to_field_ids(
    # hex_shp_pth = r"Q:\FORLand\Field_farm_size_relationship\data\vector\grid\hexagon_grid_ALL_15km_with_values.shp",
    # iacs_pth = r"Q:\FORLand\Field_farm_size_relationship\data\vector\IACS\IACS_ALL_2018_with_grassland_recl_3035.shp",
    # out_pth = r"Q:\FORLand\Field_farm_size_relationship\data\tables\field_ids_w_hexa_ids_15km_25832.csv"
    # )

    ## Reduce Bayes table to necessary columns only. Save to vector and table file
    ## Preliminary run:
    # bay_pred = gpd.read_file(r"data\predictions_bayes\preds_full_sp.gpkg")
    # bay_pred_red = bay_pred[["field_id", "farm_id.x", "farm_size_ha", "preds", "resid"]] # "geometry"
    # bay_pred_red.columns = ["field_id", "farm_id", "farm_size_ha", "preds", "resid"] # "geometry"
    # bay_pred_red.to_file(r"models\bayes_preds_temp.gpkg", driver="GPKG")
    # bay_pred_red.to_csv(r"models\bayes_preds_temp.csv")
    ## Read Bayes predictions and get the test fields
    # bay_pred_red = gpd.read_file(r"models\bayes_preds_temp.gpkg")
    # bay_pred_red = pd.read_csv(r"models\bayes_preds_temp.csv")
    # bay_train = pd.read_csv(r"data\tables\sample_large_025.csv")
    # bay_pred_test = bay_pred_red.loc[~bay_pred_red["field_id"].isin(list(bay_train["field_id"]))].copy()

    ## Run without SQR:
    # bay_pred = pd.read_csv(r"models\preds_full_bayes_w_SQR.csv")
    # bay_pred_red = bay_pred[["field_id", "preds", "resid"]]
    # bay_train = pd.read_csv(r"data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv")
    # bay_pred_test = bay_pred_red.loc[~bay_pred_red["field_id"].isin(list(bay_train["field_id"]))].copy()
    # print(f"{len(bay_pred_test)} observations for Bayes model.")
    #
    # ## Read the different versions of the RFR predictions and get the test fields
    # ## 1 - RFR run with only the selected variables
    # rfr_pred = pd.read_csv(r"models\rfr_final_run_selected_variables\rfr_predictions_rfr_final_run_selected_variables_n1000_t07.csv")
    # rfr_train = pd.read_csv(r"models\rfr_final_run_selected_variables\training_sample_field_ids.csv")
    # rfr_pred_test = rfr_pred.loc[~rfr_pred["field_id"].isin(list(rfr_train["field_id"]))].copy()
    # print(f"{len(bay_pred_test)} observations for RFR - selected variables.")
    #
    # ## 2 - RFR run with all variables
    # rfr_pred2 = pd.read_csv(r"models\rfr_final_run_all_variables\rfr_predictions_rfr_final_run_all_variables_n1000_t07.csv")
    # rfr_train2 = pd.read_csv(r"models\rfr_final_run_all_variables\training_sample_field_ids.csv")
    # rfr_pred_test2 = rfr_pred2.loc[~rfr_pred2["field_id"].isin(list(rfr_train2["field_id"]))].copy()
    # print(f"{len(bay_pred_test)} observations for RFR - all variables.")
    #
    # ## Read the reference values
    # ref = pd.read_csv(r"data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv")
    #
    # ## Merge the predictions with the reference values. Do it for all models, for a reason I can't
    # ## recall currently.
    # bay_pred_test = pd.merge(bay_pred_test[["field_id",  "preds"]],
    #                          ref[["field_id", "field_size", "new_ID_KTYP", "federal_st", "farm_id", "farm_size_r", "field_size"]], how="left", on="field_id")
    # bay_pred_test.rename(columns={"preds": "bayes_pred", "farm_size_r": "farm_size_r_bayes"}, inplace=True)
    #
    # rfr_pred_test = pd.merge(rfr_pred_test[["field_id", "rfr_pred"]],
    #                          ref[["field_id", "farm_size_r", "field_size"]], how="left", on="field_id")
    # rfr_pred_test.rename(columns={"rfr_pred": "rfr_pred_selvar", "farm_size_r": "farm_size_r_selvar"}, inplace=True)
    #
    # rfr_pred_test2 = pd.merge(rfr_pred_test2[["field_id", "rfr_pred"]],
    #                          ref[["field_id", "farm_size_r",  "field_size"]],"left", "field_id")
    # rfr_pred_test2.rename(columns={"rfr_pred": "rfr_pred_allvar", "farm_size_r": "farm_size_r_allvar"}, inplace=True)
    #
    # preds_test = pd.merge(rfr_pred_test[["field_id", "farm_size_r_selvar", "rfr_pred_selvar"]],
    #                  bay_pred_test[["field_id", "farm_id", "farm_size_r_bayes", "bayes_pred",  "field_size", "new_ID_KTYP", "federal_st"]],
    #                       left_on="field_id", right_on="field_id", how="outer")
    # preds_test = pd.merge(preds_test, rfr_pred_test2[["field_id", "farm_size_r_allvar", "rfr_pred_allvar"]],
    #                       left_on="field_id", right_on="field_id", how="outer")
    # preds_test.to_csv(r"models\predictions_bayes+rfr_selected+rfr_all_w_SQR.csv", index=False)

    ## Plot Stuff
    ## Read the prepared predictions table, if script is not completely run
    preds_test = pd.read_csv(r"models\predictions_bayes+rfr_selected+rfr_all_w_SQR.csv")

    ## Subset to make it comparable to run with SQR, which has fewer observations
    ref = pd.read_csv(rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv')
    ref.dropna(inplace=True)
    preds_test = preds_test.loc[preds_test["field_id"].isin(list(ref["field_id"]))].copy()

    ## APU plot, comparison of scatter plots, comparison of spatial maps
    # plot_apu(
    #     preds=preds_test,
    #     pred_cols=["bayes_pred", "rfr_pred_selvar", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_selvar", "farm_size_r_allvar"],
    #     subtitles=["a)", "b)", "c)"],
    #     apu_plot_pth=r"models\apu_plot_bayes+rfr_selected+rfr_all_w_sqr.png")
    #
    # plot_apu(
    #     preds=preds_test,
    #     pred_cols=["bayes_pred", "rfr_pred_selvar", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_selvar", "farm_size_r_allvar"],
    #     subtitles=["a)", "b)", "c)"],
    #     qcut_col="field_size",
    #     apu_plot_pth=r"models\apu_plot_bayes+rfr_selected+rfr_all_w_sqr_field_qcut.png")
    #
    # plot_apu(
    #     preds=preds_test,
    #     pred_cols=["bayes_pred", "rfr_pred_selvar", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_selvar", "farm_size_r_allvar"],
    #     subtitles=["a)", "b)", "c)"],
    #     cat_col="federal_st",
    #     apu_plot_pth=r"models\apu_plot_bayes+rfr_selected+rfr_all_w_sqr_federal_st.png")
    #
    # plot_apu(
    #     preds=preds_test,
    #     pred_cols=["bayes_pred", "rfr_pred_selvar", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_selvar", "farm_size_r_allvar"],
    #     subtitles=["a)", "b)", "c)"],
    #     cat_col="new_ID_KTYP",
    #     apu_plot_pth=r"models\apu_plot_bayes+rfr_selected+rfr_all_w_sqr_new_ID_KTYP.png")
    #
    # # ## Comparison of scatter plots
    # accuracy_assessment(
    #     df=preds_test,
    #     pred_cols=["bayes_pred", "rfr_pred_selvar", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_selvar", "farm_size_r_allvar"],
    #     out_pth=r"models\scatter_plot_bayes+rfr_selected+rfr_all_w_sqr.png")

    ## Comparison of spatial maps
    ## Load the hexagons and the assignment table field-ids-to-hexagons
    hex_shp = gpd.read_file(r"data\vector\grid\hexagon_grid_ALL_15km_with_values.shp")
    f_to_h = pd.read_csv(r"data\tables\field_ids_w_hexa_ids_15km_25832.csv")

    gdf_2 = gpd.read_file(fr"data\vector\administrative\GER_bundeslaender.shp")
    gdf_3 = gpd.read_file(fr"data\vector\administrative\vg-hist.utm32s.shape\daten\utm32s\shape\VG-Hist_1989-12-31_STA.shp")

    # plot_aggregated_farm_sizes_hexagon(
    #     preds=preds_test,
    #     hex_shp=hex_shp,
    #     f_to_h=f_to_h,
    #     pred_cols=["bayes_pred", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_allvar"],
    #     out_pth=r"models\map_aggregated_farm_sizes_15km_bayes+rfr_selected+rfr_all_w_sqr.png",
    #     gdf_2=gdf_2,
    #     gdf_3=gdf_3
    # )

    plot_aggregated_farm_sizes_hexagon_divided(
        preds=preds_test,
        hex_shp=hex_shp,
        f_to_h=f_to_h,
        pred_cols=["bayes_pred", "rfr_pred_allvar"],
        ref_cols=["farm_size_r_bayes", "farm_size_r_allvar"],
        out_pth=r"models\map_aggregated_farm_sizes_15km_bayes+rfr_selected+rfr_all_divided_w_sqr.png",
        gdf_2=gdf_2,
        gdf_3=gdf_3
    )

    # plot_aggregated_farm_sizes_hexagon_divided_std(
    #     preds=preds_test,
    #     hex_shp=hex_shp,
    #     f_to_h=f_to_h,
    #     pred_cols=["bayes_pred", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_r_bayes", "farm_size_r_allvar"],
    #     out_pth=r"models\map_aggregated_farm_sizes_15km_bayes+rfr_selected+rfr_all_divided_std_w_sqr.png",
    #     gdf_2=gdf_2,
    #     gdf_3=gdf_3
    # )

    # plot_aggregated_farm_sizes_hexagon_final(
    #     preds=preds_test,
    #     hex_shp=hex_shp,
    #     f_to_h=f_to_h,
    #     pred_cols=["bayes_pred", "rfr_pred_allvar"],
    #     ref_cols=["farm_size_ha", "farm_size_r_allvar"],
    #     out_pth=r"models\map_aggregated_farm_sizes_15km_bayes+rfr_selected+rfr_all_final.png",
    #     gdf_2=gdf_2,
    #     gdf_3=gdf_3,
    # )

    # bay_pred_train = bay_pred_red.loc[bay_pred_red["field_id"].isin(list(bay_train["field_id"]))].copy()
    # rfr_pred_train = rfr_pred.loc[rfr_pred["field_id"].isin(list(rfr_train["field_id"]))].copy()
    #
    # preds_train = pd.merge(rfr_pred_train[["field_id", "farm_size", "rfr_pred"]],
    #                       bay_pred_train[["field_id", "farm_id", "farm_size_ha", "preds"]], left_on="field_id",
    #                       right_on="field_id", how="outer")
    # preds_train.rename(columns={"preds": "bayes_pred"}, inplace=True)
    # preds_train.to_csv(r"models\rfr_comp_best_bayes_model_replicated\rfr+bayes_train_predictions.csv", index=False)
    #
    # plot_apu(
    #     preds=preds_train,
    #     pred_cols=["rfr_pred", "bayes_pred"],
    #     ref_cols=["farm_size", "farm_size_ha"],
    #     apu_plot_pth=r"models\rfr_comp_best_bayes_model_replicated\apu_plot_rfr+bayes_train.png")
    #
    # accuracy_assessment(
    #     df=preds_train,
    #     pred_cols=["rfr_pred", "bayes_pred"],
    #     ref_cols=["farm_size", "farm_size_ha"],
    #     out_pth=r"models\rfr_comp_best_bayes_model_replicated\scatter_plot_rfr+bayes_train.png")
    #
    # plot_aggregated_farm_sizes_hexagon(
    #     preds=preds_train,
    #     hex_shp=hex_shp,
    #     f_to_h=f_to_h,
    #     pred_cols=["rfr_pred", "bayes_pred"],
    #     ref_cols=["farm_size", "farm_size_ha"],
    #     out_pth=r"models\rfr_comp_best_bayes_model_replicated\aggregated_farm_sizes_15km_map_train.png")

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()