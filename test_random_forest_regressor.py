## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
# import shap

import numpy as np

# project library for plotting
import plotting_lib
## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"
IACS_PTH = r"data\test_run2\all_predictors\all_predictors.shp"
SAMPLE_PTH = r"data\vector\final\matched_sample-v2.shp"

## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#


def train_random_forest_regressor(X, y, n_estimators, scaler=True):

    print("Fit regressor")

    ## Design pipeline to build the treatment estimator. It standardizes the data and applies a logistic classifier
    if scaler:
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),  # test Standard Scaler
            ('regressor', RandomForestRegressor(n_estimators=n_estimators))
        ])
    else:
        pipe = Pipeline([
            ('regressor', RandomForestRegressor(n_estimators=n_estimators))
        ])

    ## Fit the classifier to the data
    pipe.fit(X, y)

    return pipe

def get_feature_importances(pipe, X_test, y_test, out_folder, descr):

    print("Determine feature importance.")
    ## Get feature importance
    ## https://mljar.com/blog/feature-importance-in-random-forest/
    sorted_idx = pipe["regressor"].feature_importances_.argsort()
    plt.barh(X_test.columns[sorted_idx], pipe["regressor"].feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.savefig(rf"{out_folder}\feature_importance{descr}.png")
    plt.close()

    # ## Get permutation feature importance
    # perm_importance = permutation_importance(pipe["regressor"], X_test.values, y_test)
    # sorted_idx = perm_importance.importances_mean.argsort()
    # plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    # plt.xlabel("Permutation Importance")
    # plt.savefig(rf"{out_folder}\permutation_feature_importance{descr}.png")
    # plt.close()

    # ## Feature Importance Computed with SHAP Values
    # explainer = shap.TreeExplainer(pipe["regressor"])
    # shap_values = explainer.shap_values(X_test.values)
    # shap.summary_plot(shap_values, X_test.values, plot_type="bar")
    # plt.savefig(rf"{out_folder}\SHAP_feature_importance{descr}.png")
    # plt.close()


def accuracy_assessment(pipe, X_test, y_test, out_pth):

    print("Accuracy assessment.")
    predictions = pipe.predict(X_test)

    ## Get R^2
    r2 = pipe.score(X_test, y_test)
    errors = abs(predictions - y_test)
    mae = round(np.mean(errors), 2)
    mape = (errors / y_test) * 100
    accuracy = 100 - np.mean(mape)

    out_dict = {"r2": r2,
                "mae": mae,
                "accuracy": accuracy}

    sns.scatterplot(y=predictions, x=y_test, s=1)
    plt.ylabel("Farm size prediction")
    plt.xlabel("Farm size reference")
    plt.savefig(out_pth)
    plt.close()

    return out_dict


def random_forest_wrapper(iacs_pth, add_vars_pth):
    ########################################## Prepare data ##########################################
    print("Read Input data.")
    iacs = gpd.read_file(iacs_pth)
    add_vars = pd.read_csv(add_vars_pth)

    print("Prepare data")
    iacs = pd.merge(iacs, add_vars, how="left", on="field_id")
    iacs["state"] = iacs["field_id"].apply(lambda x: x[:2])
    iacs["fd"] = iacs["CstMaj"].apply(lambda x: str(int(x))[1] if x != 0 else 0)
    iacs["temp_div_agg"] = 1
    iacs.loc[iacs["fd"].isin(["1", "2", "3"]), "temp_div_agg"] = "1"
    iacs.loc[iacs["fd"].isin(["4", "5", "6"]), "temp_div_agg"] = "2"
    iacs.loc[iacs["fd"].isin(["7", "8", "9"]), "temp_div_agg"] = "3"

    # log_field_size * surrf_mean_log + log_field_size * proportAgr + (1 | state) +
    # sdElevatio + temp_div_agg + SQRAvrg_log

    iacs["log_farm_size"] = np.log(iacs["farm_size"])
    iacs["log_field_size"] = np.log(iacs["fieldSizeM"])
    iacs["log_surrf_mean"] = np.log(iacs["surrf_mean"])
    iacs["log_soil_quality"] = np.log(iacs["_mean"])
    iacs["inter_lfs_lsm"] = iacs["log_field_size"] * iacs["log_surrf_mean"]
    iacs["inter_lfs_pagr"] = iacs["log_field_size"] * iacs["proportAgr"]

    ## Define relevant variables
    dep_var = "log_farm_size"
    crop_var = "temp_div_agg" ## choose between crop class or functional diverstiy - ID_KTYP or fd or temp_div_agg
    indep_vars = ["inter_lfs_lsm", "inter_lfs_pagr", 'state', 'sdElevatio', crop_var, 'log_soil_quality']
    # indep_vars = [crop_var, 'fieldSizeM', 'proportAgr', 'SlopeAvrg', 'ElevationA', 'FormerDDRm', '_mean',
    #               "surrf_mean", "surrf_median", "surrf_std", "surrf_min", "surrf_max", "surrf_no_fields"]

    for col in indep_vars:
        sub = iacs.loc[iacs[col].isna()].copy()
        if not sub.empty:
            print(f"Column {col} has {len(sub)} NAs.")

    iacs.dropna(inplace=True)

    # ########################################## Random sampling on field IDs ##########################################
    # print("Random sampling on field IDs")
    # out_folder = r"Q:\FORLand\Field_farm_size_relationship\figures\rfr_field_ids"
    # ## Exclude unnecessary columns
    # df_data = iacs[[dep_var] + indep_vars].copy()
    # ## Separate dependent and independent variables
    # y = df_data[dep_var]
    # X = df_data.loc[:, df_data.columns != dep_var]
    #
    # ## Convert categorical variables into dummy variables
    # X_encoded = pd.get_dummies(
    #     data=X,
    #     columns=["FormerDDRm", crop_var], drop_first=False)
    #
    # print("Grid search n_estimators, training size")
    # df_lst = []
    # for n_estimators in [500, 1000, 1500]:
    #     for train_size in [0.02, 0.04, 0.06, 0.08, 0.1]:
    #         print("n_estimators:", n_estimators, "training size:", train_size)
    #         descr = f"_n{n_estimators}_t{train_size}".replace('.', '')
    #
    #         test_size = 0.1
    #         ## Separate test and training data
    #         X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size)
    #
    #         pipe = train_random_forest_regressor(X_train, y_train, n_estimators)
    #
    #         # get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)
    #
    #         out_pth = f"{out_folder}\scatter_pred_vs_ref_farm_size_{descr}.png"
    #         acc_dict = accuracy_assessment(pipe, X_test, y_test, out_pth)
    #         acc_dict["n_estimators"] = n_estimators
    #         acc_dict["train_size"] = train_size
    #         acc_dict["train_num"] = len(y_train)
    #         acc_dict["test_num"] = len(y_test)
    #
    #         df = pd.DataFrame.from_dict(acc_dict, orient="index")
    #         df.columns = [descr]
    #         df_lst.append(df)
    #
    #         # ## Calculate predictions
    #         # print("Calculate predictions.")
    #         # predictions = pipe.predict(X_encoded)
    #         #
    #         # iacs["rfr_pred"] = predictions
    #         # iacs["errors"] = iacs["farm_size"] - iacs["rfr_pred"]
    #         # iacs = iacs[["field_id", "farm_size", "rfr_pred", "errors", "geometry"]]
    #         # iacs.to_file(fr"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr_predictions_{descr}.shp")
    #
    # df_out = pd.concat(df_lst, axis=1)
    # df_out.to_csv(f"{out_folder}\grid_search_field_ids.csv")

    # ########################################## Random sampling on farm IDs ##########################################
    # print("Random sampling on farm IDs")
    # out_folder = r"Q:\FORLand\Field_farm_size_relationship\figures\rfr_farm_ids"
    # n_estimators = 1000
    #
    # df_lst = []
    # print("Grid search n_estimators, training size")
    # for num_farms in [50, 100, 500, 1000, 5000, 10000]:
    #     print("No. farms used for sampling:", num_farms)
    #
    #     descr = f"_n{n_estimators}_f{num_farms}"
    #     ## Exclude unnecessary columns
    #     df_data = iacs[["farm_id"] + [dep_var] + indep_vars].copy()
    #
    #     farm_ids = list(df_data["farm_id"].unique())
    #     rand_ids = random.choices(farm_ids, k=num_farms)
    #
    #     df_train = df_data.loc[df_data["farm_id"].isin(rand_ids),  [dep_var] + indep_vars].copy()
    #     df_test = df_data.loc[~df_data["farm_id"].isin(rand_ids),  [dep_var] + indep_vars].copy()
    #     df_test = df_test.sample(n=10000)
    #
    #     ## Separate dependent and independent variables
    #     y_train = df_train[dep_var]
    #     y_test = df_test[dep_var]
    #
    #     X = df_data.loc[:, df_data.columns != dep_var]
    #     X_train_orig = df_train.loc[:, df_train.columns != dep_var]
    #     X_test_orig = df_test.loc[:, df_test.columns != dep_var]
    #
    #     ## Convert categorical variables into dummy variables
    #     X_encoded = pd.get_dummies(
    #         data=X,
    #         columns=["FormerDDRm", crop_var], drop_first=False)
    #
    #     X_train = pd.get_dummies(
    #         data=X_train_orig,
    #         columns=["FormerDDRm", crop_var], drop_first=False)
    #
    #     X_test = pd.get_dummies(
    #         data=X_test_orig,
    #         columns=["FormerDDRm", crop_var], drop_first=False)
    #     ## Due to special way of splitting training and test data, it is possible that some columns occur in the
    #     ## training or test dataset that do not occur in the other, because the dummy columns are not generate as a
    #     ## certain value does not occur in the respective dataset.
    #     train_cols_miss = [col for col in X_test.columns if col not in X_train.columns]
    #     if len(train_cols_miss) > 0:
    #         for col in train_cols_miss:
    #             X_train[col] = 0
    #         X_train = X_train[X_test.columns]
    #
    #     test_cols_miss = [col for col in X_train.columns if col not in X_test.columns]
    #     if len(test_cols_miss) > 0:
    #         for col in test_cols_miss:
    #             X_test[col] = 0
    #         X_test = X_test[X_train.columns]
    #
    #     ## Fit regressor to training data.
    #     pipe = train_random_forest_regressor(X_train, y_train, n_estimators=n_estimators)
    #
    #     # get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)
    #     out_pth = f"{out_folder}\scatter_pred_vs_ref_farm_size_{descr}.png"
    #     acc_dict = accuracy_assessment(pipe, X_test, y_test, out_pth)
    #     acc_dict["n_estimators"] = n_estimators
    #     acc_dict["num_farms"] = num_farms
    #     acc_dict["train_num"] = len(y_train)
    #     acc_dict["test_num"] = len(y_test)
    #
    #     df = pd.DataFrame.from_dict(acc_dict, orient="index")
    #     df.columns = [descr]
    #     df_lst.append(df)
    #
    #     # ## Calculate predictions
    #     # print("Calculate predictions.")
    #     # predictions = pipe.predict(X_encoded)
    #     #
    #     # iacs["rfr_pred"] = predictions
    #     # iacs = iacs[["field_id", "farm_size", "rfr_pred", "geometry"]]
    #     # iacs.to_file(fr"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr_predictions_{descr}.shp")
    # df_out = pd.concat(df_lst, axis=1)
    # df_out.to_csv(f"{out_folder}\grid_search_farm_ids.csv")

    # ########################################## Testing separate federal states #######################################
    # print("Testing separate federal states")
    # out_folder = r"Q:\FORLand\Field_farm_size_relationship\figures\rfr_separate_states"
    # n_estimators = 1000
    # train_size = 0.1
    # test_size = 0.1
    #
    # df_lst = []
    # for state in ["BB", "LS"]: #, "TH",  "SA", "BV"
    #     print("n_estimators:", n_estimators, "training size:", train_size)
    #     descr = f"_s{state}_n{n_estimators}_t{train_size}_{crop_var}_stratsampling".replace('.', '')
    #     iacs_sub = iacs.loc[iacs["state"] == state].copy()
    #
    #     ## Exclude unnecessary columns
    #     df_data = iacs_sub[[dep_var] + indep_vars].copy()
    #     ## Separate dependent and independent variables
    #     y = df_data[dep_var]
    #     X = df_data.loc[:, df_data.columns != dep_var]
    #
    #     ## Convert categorical variables into dummy variables
    #     X_encoded = pd.get_dummies(
    #         data=X,
    #         columns=["FormerDDRm", crop_var], drop_first=False)
    #     strata = pd.qcut(y, q=200, labels=[str(i) for i in range(1, 201)])
    #
    #     ## Separate test and training data
    #     X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size,
    #                                                         stratify=strata)
    #
    #     pipe = train_random_forest_regressor(X_train, y_train, n_estimators)
    #
    #     get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)
    #
    #     out_pth = f"{out_folder}\scatter_pred_vs_ref_farm_size{descr}.png"
    #     acc_dict = accuracy_assessment(pipe, X_test, y_test, out_pth)
    #     acc_dict["n_estimators"] = n_estimators
    #     acc_dict["train_size"] = train_size
    #     acc_dict["train_num"] = len(y_train)
    #     acc_dict["test_num"] = len(y_test)
    #     acc_dict["federal_state"] = state
    #
    #     df = pd.DataFrame.from_dict(acc_dict, orient="index")
    #     df.columns = [descr]
    #     df_lst.append(df)
    #
    #     ## Calculate predictions
    #     print("Calculate predictions.")
    #     predictions = pipe.predict(X_encoded)
    #
    #     iacs_sub["rfr_pred"] = predictions
    #     iacs_sub["errors"] = iacs_sub["farm_size"] - iacs_sub["rfr_pred"]
    #     iacs_sub = iacs_sub[["field_id", "farm_size", "rfr_pred", "errors", "geometry"]]
    #     iacs_sub.to_file(fr"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr_predictions_{descr}.shp")
    #
    # df_out = pd.concat(df_lst, axis=1)
    # df_out.to_csv(f"{out_folder}\grid_search_field_ids_per_state{descr}.csv")

    ## How does the SQR impact the predictions?
    ## How does a stratified sampling impact the predictions?
    ## How does the full model compare to a regionalized model?

    # ########################################## Final model with random sampling on field IDs #########################
    print("FINAL MODEL with random sampling on field IDs")
    out_folder = r"Q:\FORLand\Field_farm_size_relationship\figures\rfr_final_model"
    n_estimators = 1000
    train_size = 0.1
    print("n_estimators:", n_estimators, "training size:", train_size)
    descr = f"_for_bayes_comparison_n{n_estimators}_t{train_size}".replace('.', '')

    ## Exclude unnecessary columns
    df_data = iacs[[dep_var] + indep_vars].copy()
    ## Separate dependent and independent variables
    y = df_data[dep_var]
    X = df_data.loc[:, df_data.columns != dep_var]

    ## Convert categorical variables into dummy variables
    X_encoded = pd.get_dummies(
        data=X,
        columns=["state", crop_var], drop_first=False)

    df_lst = []

    test_size = 0.1
    ## Separate test and training data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size)

    pipe = train_random_forest_regressor(X_train, y_train, n_estimators, scaler=False)

    get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)

    out_pth = f"{out_folder}\scatter_pred_vs_ref_farm_size_{descr}.png"
    acc_dict = accuracy_assessment(pipe, X_test, y_test, out_pth)
    acc_dict["n_estimators"] = n_estimators
    acc_dict["train_size"] = train_size
    acc_dict["train_num"] = len(y_train)
    acc_dict["test_num"] = len(y_test)

    df = pd.DataFrame.from_dict(acc_dict, orient="index")
    df.columns = [descr]
    df_lst.append(df)

    print("Write accuracy results out.")
    df_out = pd.concat(df_lst, axis=1)
    df_out.to_csv(f"{out_folder}\grid_search_field_ids.csv")

    ## Calculate predictions
    print("Calculate predictions.")
    predictions = pipe.predict(X_encoded)

    iacs["rfr_pred_log"] = predictions
    iacs["rfr_pred"] = np.exp(predictions)
    iacs["errors"] = iacs["farm_size"] - iacs["rfr_pred"]
    iacs = iacs[["field_id", "farm_size", "log_farm_size", "rfr_pred", "rfr_pred_log", "errors", "geometry"]]

    print("Write result shapefile out.")
    iacs.to_file(fr"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr_predictions_{descr}.shp")


    # ########################################## Model on heaxgons #########################
    # out_folder = r"Q:\FORLand\Field_farm_size_relationship\figures\rfr_hexagons"
    # n_estimators = 1000
    # train_size = 0.1
    # test_size = 0.1
    #
    # print("n_estimators:", n_estimators, "training size:", train_size)
    # descr = f"_hexagons_n{n_estimators}_t{train_size}".replace('.', '')
    #
    # def mode(aggr_series):
    #     return pd.Series.mode(aggr_series).iloc[0]
    #
    # iacs_hexa = iacs.groupby("hexa_id").agg(
    #     avg_farm_size=pd.NamedAgg("farm_size", "mean"),
    #     avg_field_size=pd.NamedAgg("fieldSizeM", "mean"),
    #     sd_field_size=pd.NamedAgg("fieldSizeM", np.std),
    #     min_field_size=pd.NamedAgg("fieldSizeM", "min"),
    #     max_field_size=pd.NamedAgg("fieldSizeM", "max"),
    #     major_crop=pd.NamedAgg("ID_KTYP", mode),
    #     # major_fd=pd.NamedAgg("fd", mode),
    #     former_gdr=pd.NamedAgg("FormerDDRm", mode),
    #     share_agric=pd.NamedAgg("proportAgr", "first"),
    #     avg_slope=pd.NamedAgg("SlopeAvrg", "mean"),
    #     avg_elevation=pd.NamedAgg("ElevationA", "mean"),
    #     avg_rugged=pd.NamedAgg("sdElevatio", "mean"),
    #     avg_sqr=pd.NamedAgg("_mean", "mean")
    # ).reset_index()
    # iacs_hexa.dropna(inplace=True)
    #
    # dep_var = "avg_farm_size"
    # crop_var = "major_crop"
    # indep_vars = ["avg_field_size", "sd_field_size", "min_field_size", "max_field_size", crop_var, #"former_gdr",
    #               "share_agric", "avg_slope", "avg_elevation", "avg_rugged", "avg_sqr"]
    #
    # ## Exclude unnecessary columns
    # df_data = iacs_hexa[[dep_var] + indep_vars].copy()
    # ## Separate dependent and independent variables
    # y = df_data[dep_var]
    # X = df_data.loc[:, df_data.columns != dep_var]
    #
    # ## Convert categorical variables into dummy variables
    # X_encoded = pd.get_dummies(
    #     data=X,
    #     columns=[crop_var], drop_first=False) #"former_gdr",
    #
    # df_lst = []
    #
    # ## Separate test and training data
    # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size)
    #
    # pipe = train_random_forest_regressor(X_train, y_train, n_estimators)
    #
    # get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)
    #
    # out_pth = f"{out_folder}\scatter_pred_vs_ref_farm_size_{descr}.png"
    # acc_dict = accuracy_assessment(pipe, X_test, y_test, out_pth)
    # acc_dict["n_estimators"] = n_estimators
    # acc_dict["train_size"] = train_size
    # acc_dict["train_num"] = len(y_train)
    # acc_dict["test_num"] = len(y_test)
    #
    # df = pd.DataFrame.from_dict(acc_dict, orient="index")
    # df.columns = [descr]
    # df_lst.append(df)
    #
    # ## Calculate predictions
    # print("Calculate predictions.")
    # predictions = pipe.predict(X_encoded)
    #
    # iacs_hexa["rfr_pred"] = predictions
    # iacs_hexa["errors"] = iacs_hexa[dep_var] - iacs_hexa["rfr_pred"]
    # iacs_hexa = iacs_hexa[["hexa_id", dep_var, "rfr_pred", "errors"]]
    # hexa_shp = gpd.read_file(r"Q:\FORLand\Field_farm_size_relationship\data\vector\grid\hexagon_grid_ALL_5km_with_values.shp")
    # hexa_shp = hexa_shp[["hexa_id", "geometry"]].copy()
    # hexa_shp = pd.merge(hexa_shp, iacs_hexa, how="left", on="hexa_id")
    # hexa_shp.to_file(fr"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr_predictions_hexa_5km_{descr}.shp")
    #
    # df_out = pd.concat(df_lst, axis=1)
    # df_out.to_csv(f"{out_folder}\grid_search_field_ids{descr}.csv")


def plot_apu(preds, pred_cols, ref_col, apu_plot_pth):
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

    ## Discretize farm sizes into ventiles
    # pred["bins"] = pd.cut(pred[ref_col], bins=20) #labels=range(1, 21)
    preds["bins"] = pd.qcut(preds[ref_col], q=20)  #

    fig, axs = plt.subplots(nrows=len(pred_cols), ncols=1, figsize=plotting_lib.cm2inch(30, 15*len(pred_cols)))

    for i, pred_col in enumerate(pred_cols):
        res_dict = {"bin": [], "n": [], "a": [], "p": [], "u": []}
        for bin in preds["bins"].unique():
            df_sub = preds.loc[preds["bins"] == bin].copy()
            n = len(df_sub)
            res = df_sub[pred_col] - df_sub[ref_col]
            a = np.sum(res)/n
            p = np.sqrt(np.sum((res-a)*(res-a))/(n-1))
            u = np.sqrt(np.sum(res*res)/n)
            res_dict["bin"].append(bin)
            res_dict["n"].append(n)
            res_dict["a"].append(a)
            res_dict["p"].append(p)
            res_dict["u"].append(u)
    
        apu_df = pd.DataFrame.from_dict(res_dict)
        apu_df.sort_values(by="bin", inplace=True)
        apu_df.index = range(len(apu_df))
    
        n = len(preds)
        res = preds[pred_col] - preds[ref_col]
        a = np.sum(res) / n
        p = np.sqrt(np.sum((res - a) * (res - a)) / (n - 1))
        u = np.sqrt(np.sum(res * res) / n)

        if len(pred_cols) == 1:
            ax = axs
        else:
            ax = axs[i]

        ax.set_title(pred_col)
    
        sns.lineplot(data=apu_df["a"], marker='o', sort=False, ax=ax, color="blue")
        sns.lineplot(data=apu_df["p"], marker='o', sort=False, ax=ax, color="orange")
        sns.lineplot(data=apu_df["u"], marker='o', sort=False, ax=ax, color="green")
        legend_elements = [Patch(facecolor="blue", edgecolor=None, label="Accuracy"),
                           Patch(facecolor="orange", edgecolor=None, label="Precision"),
                           Patch(facecolor="green", edgecolor=None, label="Uncertainty")]
        ax.legend(handles=legend_elements, bbox_to_anchor=(.7, .95), ncol=3, title="Legend")
        ax.set_ylabel('A/P/U')
        ax.set_xlabel('farm sizes [ha]')

        text = f"{n} farms\nAccuracy: {round(a,3)}\nPrecision: {round(p,3)}\nUncertainty: {round(u,3)}"
        ax.annotate(s=text, xy=(.1, .8), xycoords="axes fraction")
        ax.tick_params(axis='x', labelrotation=45)

        ax2 = ax.twinx()
        ax2.set_xlabel('Number of farms')
        sns.barplot(data=apu_df, x='bin', y='n', alpha=0.5, ax=ax2)

        fig.tight_layout()

    plt.savefig(apu_plot_pth)
    plt.close()

    print("Plotting done!")

## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    # random_forest_wrapper(
    #     iacs_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\all_predictors_sqr.shp",
    #     add_vars_pth=r"Q:\FORLand\Field_farm_size_relationship\data\tables\surrounding_fields_stats_ALL.csv")
    #
    rfr_pred = gpd.read_file(r"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr_predictions__for_bayes_comparison_n1000_t01.shp")
    bay_pred = gpd.read_file(r"Q:\FORLand\Field_farm_size_relationship\data\predictions_bayes\preds_full_sp.shp")

    preds = pd.merge(rfr_pred[["field_id", "farm_size", "rfr_pred"]],
                       bay_pred[["field_d", "farm_id", "preds"]], left_on="field_id", right_on="field_d", how="left")
    preds.dropna(subset=["preds"], inplace=True)
    preds.dropna(subset=["rfr_pred"], inplace=True)
    preds.rename(columns={"preds": "bayes_pred"}, inplace=True)
    # preds.to_csv(r"Q:\FORLand\Field_farm_size_relationship\data\vector\rfr+bayes_predictions.csv", index=False)

    # plot_apu(
    #     preds=preds,
    #     pred_cols=["rfr_pred"],
    #     ref_col="farm_size",
    #     apu_plot_pth=r"Q:\FORLand\Field_farm_size_relationship\figures\apu_plot_rfr.png")
    #
    # plot_apu(
    #     preds=preds,
    #     pred_cols=["bayes_pred"],
    #     ref_col="farm_size",
    #     apu_plot_pth=r"Q:\FORLand\Field_farm_size_relationship\figures\apu_plot_bayes.png")

    plot_apu(
        preds=preds,
        pred_cols=["rfr_pred", "bayes_pred"],
        ref_col="farm_size",
        apu_plot_pth=r"Q:\FORLand\Field_farm_size_relationship\figures\apu_plot_rfr+bayes.png")

    ## Bayes predictions
    # plot_apu(
    #     pred_pth=r"Q:\FORLand\Field_farm_size_relationship\data\predictions_bayes\preds_full_sp.shp",
    #     pred_col="preds",
    #     ref_col="farm_sz",
    #     apu_plot_pth=r"Q:\FORLand\Field_farm_size_relationship\figures\apu_plot_bayes.png")

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
