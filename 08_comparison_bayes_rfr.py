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

def plot_apu(preds, pred_cols, ref_cols, apu_plot_pth, subtitles=None, qcut=None):
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
        if qcut:
            preds_curr["bins"] = pd.qcut(preds_curr[ref_col], q=qcut)
        else:
            nbins = 20
            bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500),
                                                 (500, 1000), (1000, 7000)])
            preds_curr["bins"] = pd.cut(preds_curr[ref_col], bins=bins,) #, labels=labels)
            t = preds_curr.loc[preds_curr["bins"].isna()].copy()
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
        sns.lineplot(data=apu_df["a"], marker='o', sort=False, ax=ax, color="blue", zorder=1)
        sns.lineplot(data=apu_df["p"], marker='o', sort=False, ax=ax, color="orange", zorder=1)
        sns.lineplot(data=apu_df["u"], marker='o', sort=False, ax=ax, color="green", zorder=1)
        sns.lineplot(data=apu_df["mape"], marker='o', sort=False, ax=ax, color="grey")

        ax.set_ylabel('ME/RMSE/MAPE/P')

        text = f"{n:,} fields\nME: {round(a, 1)}\nRMSE: {round(u, 1)}\nMAPE: {round(mape, 1)}\n?: {round(p, 1)}"
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


def plot_aggregated_farm_sizes_hexagon(preds, hex_shp, f_to_h, pred_cols, ref_cols, out_pth):

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=3, figsize=plotting_lib.cm2inch(30, 10 * len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        ref_col = ref_cols[i]

        df_curr = preds.dropna(subset=[pred_col]).copy()
        df_comb = pd.merge(df_curr, f_to_h, how="left", on="field_id")
        df_plt = df_comb.groupby("hexa_id").agg(
            agg_pred=pd.NamedAgg(pred_col, "mean"),
            agg_ref=pd.NamedAgg(ref_col, "mean")
        ).reset_index()

        gdf = pd.merge(hex_shp, df_plt, how="left", on="hexa_id")
        gdf.dropna(inplace=True)

        # Get the range of values for colormap normalization
        vmin = min(gdf["agg_ref"].min(), gdf["agg_pred"].min())
        vmax = max(gdf["agg_ref"].max(), gdf["agg_pred"].max())

        # Plot the reference and prediction maps
        gdf.plot(column="agg_ref", ax=axs[i, 0], vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(ref_col)
        axs[i, 0].axis('off')
        gdf.plot(column="agg_pred", ax=axs[i, 1], vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(pred_col)
        axs[i, 1].axis('off')

        predictions = gdf["agg_pred"]
        reference = gdf["agg_ref"]

        errors = abs(predictions - reference)
        mae = round(np.mean(errors), 2)
        mse = mean_squared_error(reference, predictions)
        rmse = math.sqrt(mse)
        mape = (errors / reference) * 100
        accuracy = 100 - np.mean(mape)

        sns.scatterplot(y=predictions, x=reference, s=1, ax=axs[i, 2])

        # Calculate the limits for the plot
        min_val = min(min(reference), min(predictions))
        max_val = max(max(reference), max(predictions))

        # Plot the 1-to-1 line
        axs[i, 2].plot([min_val, max_val], [min_val, max_val], color='grey', label='1-to-1 Line')

        axs[i, 2].set_ylabel("Farm size prediction [ha]")
        axs[i, 2].set_xlabel("Farm size reference [ha]")

        text = f"MAE: {round(mae, 1)}\nMSE: {round(mse, 1)}\nRMSE: {round(rmse, 1)}, \nAccuracy: {round(accuracy, 1)}"
        axs[i, 2].annotate(text=text, xy=(.1, .8), xycoords="axes fraction")

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

    # bay_pred = gpd.read_file(r"data\predictions_bayes\preds_full_sp.gpkg")
    # bay_pred_red = bay_pred[["field_id", "farm_id.x", "farm_size_ha", "preds", "resid"]] # "geometry"
    # bay_pred_red.columns = ["field_id", "farm_id", "farm_size_ha", "preds", "resid"] # "geometry"
    # bay_pred_red.to_file(r"models\bayes_preds_temp.gpkg", driver="GPKG")
    # bay_pred_red.to_csv(r"models\bayes_preds_temp.csv")

    # hex_shp = gpd.read_file(r"Q:\FORLand\Field_farm_size_relationship\data\vector\grid\hexagon_grid_ALL_15km_with_values.shp")
    # f_to_h = pd.read_csv(r"Q:\FORLand\Field_farm_size_relationship\data\tables\field_ids_w_hexa_ids_15km_25832.csv")
    #
    # # bay_pred_red = gpd.read_file(r"models\bayes_preds_temp.gpkg")

    # bay_pred_red = pd.read_csv(r"models\bayes_preds_temp.csv")
    # bay_train = pd.read_csv(r"data\tables\sample_large_025.csv")
    # bay_pred_test = bay_pred_red.loc[~bay_pred_red["field_id"].isin(list(bay_train["field_id"]))].copy()
    #
    # rfr_pred = pd.read_csv(r"models\rfr_comp_best_bayes_model_replicated_same_ids\rfr_predictions_rfr_comp_best_bayes_model_replicated_same_ids_n1000_t01.csv")
    # rfr_train = pd.read_csv(r"models\rfr_comp_best_bayes_model_replicated_same_ids\training_sample_field_ids.csv")
    # rfr_pred_test = rfr_pred.loc[~rfr_pred["field_id"].isin(list(rfr_train["field_id"]))].copy()
    #
    # rfr_pred2 = pd.read_csv(
    #     r"models\rfr_comp_best_bayes_model_replicated_same_ids_all_variables\rfr_predictions_rfr_comp_best_bayes_model_replicated_same_ids_all_variables_n1000_t01.csv")
    # rfr_train2 = pd.read_csv(r"models\rfr_comp_best_bayes_model_replicated_same_ids_all_variables\training_sample_field_ids.csv")
    # rfr_pred_test2 = rfr_pred2.loc[~rfr_pred2["field_id"].isin(list(rfr_train2["field_id"]))].copy()
    #
    # rfr_pred3 = pd.read_csv(
    #     r"models\rfr_all_variables\rfr_predictions_rfr_all_variables_n1000_t025.csv")
    # rfr_train3 = pd.read_csv(r"models\rfr_all_variables\training_sample_field_ids.csv")
    # rfr_pred_test3 = rfr_pred3.loc[~rfr_pred3["field_id"].isin(list(rfr_train3["field_id"]))].copy()
    #
    # ref = pd.read_csv(r"models\all_predictors_w_grassland_log_and_scaled.csv")
    # rfr_pred_test = pd.merge(rfr_pred_test[["field_id", "rfr_pred"]], ref[["field_id", "farm_size_r", "fieldCount", "field_size"]],
    #                          "left", "field_id")
    #
    # bay_pred_test = pd.merge(bay_pred_test[["field_id", "farm_id", "farm_size_ha", "preds"]],
    #                          ref[["field_id", "fieldCount", "field_size"]],"left", "field_id")
    #
    # rfr_pred_test2 = pd.merge(rfr_pred_test2[["field_id", "rfr_pred"]],
    #                          ref[["field_id", "farm_size_r", "fieldCount", "field_size"]],
    #                          "left", "field_id")
    # rfr_pred_test2.rename(columns={"rfr_pred": "rfr_pred_bcav", "farm_size_r": "farm_size_r_bcav"}, inplace=True)
    #
    # rfr_pred_test3 = pd.merge(rfr_pred_test3[["field_id", "rfr_pred"]],
    #                          ref[["field_id", "farm_size_r", "fieldCount", "field_size"]],
    #                          "left", "field_id")
    # rfr_pred_test3.rename(columns={"rfr_pred": "rfr_pred_av", "farm_size_r": "farm_size_r_av"}, inplace=True)
    #
    # # rfr_pred["rfr_pred"] = rfr_pred["rfr_pred"] / 10000
    # # rfr_pred["farm_size"] = rfr_pred["farm_size"] / 10000
    #
    # # ## Temporary:
    # # rfr_pred_test = rfr_pred_test.loc[rfr_pred_test["fieldCount"] > 1].copy()
    # # rfr_pred_test = rfr_pred_test.loc[rfr_pred_test["farm_size_r"] >= 0.3].copy()
    # #
    # # bay_pred_test = bay_pred_test.loc[bay_pred_test["fieldCount"] > 1].copy()
    # # bay_pred_test = bay_pred_test.loc[bay_pred_test["farm_size_ha"] >= 0.3].copy()
    # #
    # preds_test = pd.merge(rfr_pred_test[["field_id", "farm_size_r", "rfr_pred"]],
    #                  bay_pred_test[["field_id", "farm_id", "farm_size_ha", "preds"]],
    #                       left_on="field_id", right_on="field_id", how="outer")
    #
    # preds_test = pd.merge(preds_test, rfr_pred_test2[["field_id", "farm_size_r_bcav", "rfr_pred_bcav"]],
    #                       left_on="field_id", right_on="field_id", how="outer")
    #
    # preds_test = pd.merge(preds_test, rfr_pred_test3[["field_id", "farm_size_r_av", "rfr_pred_av"]],
    #                       left_on="field_id", right_on="field_id", how="outer")
    # # # preds.dropna(subset=["preds"], inplace=True)
    # # # preds.dropna(subset=["rfr_pred"], inplace=True)
    # preds_test.rename(columns={"preds": "bayes_pred"}, inplace= True)
    #
    # preds_test.to_csv(r"models\rfr_av+rfr_bcav+bayes_test_predictions.csv", index=False)
    preds_test = pd.read_csv(r"models\rfr_av+rfr_bcav+bayes_test_predictions.csv")

    plot_apu(
        preds=preds_test,
        pred_cols=["bayes_pred", "rfr_pred", "rfr_pred_bcav"],
        ref_cols=["farm_size_ha", "farm_size_r", "farm_size_r_bcav"],
        subtitles=["a)", "b)", "c)"],
        apu_plot_pth=r"models\rfr_all_variables\apu_plot_rfr+rfr_av+rfr_bcav+bayes_test_cut.png")

    # plot_apu(
    #     preds=preds_test,
    #     pred_cols=["rfr_pred", "bayes_pred"],
    #     ref_cols=["farm_size_r", "farm_size_ha"],
    #     apu_plot_pth=r"models\rfr_comp_best_bayes_model_replicated_same_ids\apu_plot_rfr+bayes_test.png",
    #     qcut=20
    # )

    accuracy_assessment(
        df=preds_test,
        pred_cols=["bayes_pred", "rfr_pred", "rfr_pred_bcav"],
        ref_cols=["farm_size_ha", "farm_size_r", "farm_size_r_bcav"],
        out_pth=r"models\rfr_all_variables\scatter_plot_rfr++rfr_av+rfr_bcav+bayes_test.png")

    #### Compare
    # plot_aggregated_farm_sizes_hexagon(
    #     preds=preds_test,
    #     hex_shp=hex_shp,
    #     f_to_h=f_to_h,
    #     pred_cols=["rfr_pred", "bayes_pred"],
    #     ref_cols=["farm_size", "farm_size_ha"],
    #     out_pth=r"models\rfr_comp_best_bayes_model_replicated\aggregated_farm_sizes_15km_map_test.png")

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