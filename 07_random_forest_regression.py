## Authors: 
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
import math
import random
from shapely.geometry import Point
from osgeo import ogr


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib
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


def train_random_forest_regressor(X, y, n_estimators, scaler=False, max_features="sqrt"):

    print("Fit regressor")

    ## Design pipeline to build the treatment estimator. It standardizes the data and applies a logistic classifier
    if scaler:
        pipe = Pipeline([
            ('scaler', StandardScaler()),  # alternative is min-max scaler
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_features=max_features))
        ])
    else:
        pipe = Pipeline([
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_features=max_features))
        ])

    ## Fit the classifier to the data
    pipe.fit(X, y)

    return pipe

def get_feature_importances(pipe, X_test, y_test, out_folder, descr):

    print("Determine feature importance.")
    ## Get feature importance
    ## https://mljar.com/blog/feature-importance-in-random-forest/
    sorted_idx = pipe["regressor"].feature_importances_.argsort()
    plt.figure(figsize=(16,16))
    plt.barh(X_test.columns[sorted_idx], pipe["regressor"].feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.tight_layout()
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


def accuracy_assessment(pipe, X_test, y_test, out_folder, descr):

    print("Accuracy assessment.")
    predictions = pipe.predict(X_test)

    r2 = pipe.score(X_test, y_test)
    errors = abs(predictions - y_test)
    mae = round(np.mean(errors), 2)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    mape = (errors / y_test) * 100
    accuracy = 100 - np.mean(mape)

    acc_dict = {"r2": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "accuracy": accuracy
                }

    # Calculate the limits for the plot
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))

    sns.scatterplot(y=predictions, x=y_test, s=1)
    plt.plot([min_val, max_val], [min_val, max_val], color='grey', label='1-to-1 Line')
    plt.ylabel("Prediction")
    plt.xlabel("Reference")
    text = f"MAE: {round(mae, 1)}\nRMSE: {round(rmse, 1)}\nMAPE: {round(np.mean(mape), 1)}"
    plt.annotate(text=text, xy=(.1, .8), xycoords="axes fraction")
    plt.savefig(os.path.join(out_folder, f"scatter_pred_vs_ref_{descr}.png"))
    plt.close()

    return acc_dict

def random_forest_wrapper_old(iacs_pth, add_vars_pth):
    ########################################## Prepare data ##########################################
    print("Read Input data.")
    iacs = gpd.read_file(iacs_pth)
    add_vars = pd.read_csv(add_vars_pth)

    print("Prepare data")
    iacs = pd.merge(iacs, add_vars, how="left", on="field_id")
    iacs["state"] = iacs["field_id"].apply(lambda x: x[:2])
    # iacs["fd"] = iacs["CstMaj"].apply(lambda x: str(int(x))[1] if x != 0 else 0)
    # iacs["temp_div_agg"] = 1
    # iacs.loc[iacs["fd"].isin(["1", "2", "3"]), "temp_div_agg"] = "1"
    # iacs.loc[iacs["fd"].isin(["4", "5", "6"]), "temp_div_agg"] = "2"
    # iacs.loc[iacs["fd"].isin(["7", "8", "9"]), "temp_div_agg"] = "3"

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

def random_forest_wrapper_new(iacs_pth, out_folder):
    ########################################## Prepare data ##########################################
    print("Read Input data.")
    iacs = pd.read_csv(iacs_pth)

    ## Define relevant variables
    dep_var = "log_farm_size"
    crop_var = "new_IDKTYP" ## choose between crop class or functional diverstiy - ID_KTYP or fd or temp_div_agg
    ## interaction crop var * state
    iacs["inter_cr_st"] = iacs[crop_var] + iacs["federal_st"]
    iacs["inter_sfm_prag"] = iacs["surrf_mean_log"] + iacs["propAg1000"]
    indep_vars = ["log_field_size", "surrf_mean_log", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"]
    # indep_vars = [crop_var, 'fieldSizeM', 'proportAgr', 'SlopeAvrg', 'ElevationA', 'FormerDDRm', '_mean',
    #               "surrf_mean", "surrf_median", "surrf_std", "surrf_min", "surrf_max", "surrf_no_fields"]

    for col in indep_vars:
        sub = iacs.loc[iacs[col].isna()].copy()
        if not sub.empty:
            print(f"Column {col} has {len(sub)} NAs.")

    iacs.dropna(inplace=True)

    # ########################################## Final model with random sampling on field IDs #########################
    print("FINAL MODEL with random sampling on field IDs")

    n_estimators = 1000
    train_size = 0.25
    print("n_estimators:", n_estimators, "training size:", train_size)
    descr = f"comparison_to_best_bayes_model_n{n_estimators}_t{train_size}".replace('.', '')

    ## Exclude unnecessary columns
    df_data = iacs[[dep_var] + indep_vars].copy()
    ## Separate dependent and independent variables
    y = df_data[dep_var]
    X = df_data.loc[:, df_data.columns != dep_var]

    ## Convert categorical variables into dummy variables
    X_encoded = pd.get_dummies(
        data=X,
        columns=["inter_cr_st"], drop_first=False)

    df_lst = []

    test_size = 0.1
    ## Separate test and training data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size)

    pipe = train_random_forest_regressor(X_encoded, y, n_estimators, scaler=False)

    joblib.dump(pipe, rf'{out_folder}\rfr_model_{descr}.pkl')

    get_feature_importances(pipe=pipe, X_test=X_encoded, y_test=y, out_folder=out_folder, descr=descr)

    out_pth = f"{out_folder}\scatter_pred_vs_ref_farm_size_in_sample_{descr}.png"
    acc_dict = accuracy_assessment(pipe, X_encoded, y, out_pth)
    acc_dict["n_estimators"] = n_estimators
    acc_dict["train_size"] = train_size
    acc_dict["train_num"] = len(y)
    acc_dict["test_num"] = len(y)

    df = pd.DataFrame.from_dict(acc_dict, orient="index")
    df.columns = [descr]
    df_lst.append(df)

    print("Write accuracy results out.")
    df_out = pd.concat(df_lst, axis=1)
    df_out.to_csv(fr"{out_folder}\accuracy_in_sample.csv")

    ## Calculate predictions
    print("Calculate predictions.")
    predictions = pipe.predict(X_encoded)

    iacs["rfr_pred_log"] = predictions
    iacs["rfr_pred"] = np.exp(predictions)
    iacs["errors"] = iacs["farm_size"] - iacs["rfr_pred"]
    iacs = iacs[["field_id", "farm_size", "log_farm_size", "rfr_pred", "rfr_pred_log", "errors"]]

    print("Write result shapefile out.")
    iacs.to_csv(fr"{out_folder}\rfr_predictions_{descr}.csv", index=False)

def random_forest_wrapper_clean(iacs_pth, dep_var, indep_vars, categorical_vars, run_descr, out_folder, train_ids=None,
                                test_ids=None, n_estimators=500, train_size=0.1, test_size=0.1, max_features="sqrt",
                                existing_model_pth=None, make_predictions=True, calc_feature_importance=True,
                                make_accuracy_assessment=True, write_model_out=True):

    print("Read data.")
    iacs = pd.read_csv(iacs_pth)

    ## Create output folder
    try:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    except OSError:
        print('Error: Creating directory. ' + out_folder)

    ## Define Random Forest parameters
    print("RFR parameters - n_estimators:", n_estimators, "training size:", train_size, "max. features:", max_features)
    descr = f"{run_descr}_n{n_estimators}_t{train_size}".replace('.', '')

    ## Drop other NAs
    print("Prepare data")
    for col in indep_vars:
        sub = iacs.loc[iacs[col].isna()].copy()
        if not sub.empty:
            print(f"Column {col} has {len(sub)} NAs.")

    iacs.dropna(inplace=True)

    ## Exclude unnecessary columns
    df_data = iacs[[dep_var] + indep_vars].copy()
    ## Separate dependent and independent variables
    y = df_data[dep_var]
    X = df_data.loc[:, df_data.columns != dep_var]

    ## Convert categorical variables into dummy variables
    X_encoded = pd.get_dummies(
        data=X,
        columns=categorical_vars,
        drop_first=False)

    ## Separate test and training data
    if train_ids:
        print(f"Using the provided {len(train_ids)} for training.")
        indeces = iacs.loc[iacs["field_id"].isin(train_ids)]
        indeces = list(indeces.index)
        X_train = X_encoded.loc[indeces]
        y_train = y.loc[indeces]
        if test_ids:
            indeces = iacs.loc[iacs["field_id"].isin(test_ids)]
            indeces = list(indeces.index)
            X_test = X_encoded.loc[indeces]
            y_test = y.loc[indeces]
        else:
            indeces = iacs.loc[~iacs["field_id"].isin(train_ids)]
            indeces = list(indeces.index)
            random_indeces = random.sample(indeces, int(test_size * len(indeces)))
            X_test = X_encoded.loc[random_indeces]
            y_test = y.loc[random_indeces]
        num_train = len(y_train)
        num_test = len(y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size,
                                                        random_state=42)
        num_train = len(y_train)
        num_test = len(y_test)

    df_train = iacs.loc[X_train.index]
    df_train[["field_id"]].to_csv(rf'{out_folder}\training_sample_field_ids.csv', index=False)

    ## Fit regressor
    if not existing_model_pth:
        pipe = train_random_forest_regressor(X_train, y_train, n_estimators, scaler=False, max_features=max_features)
        if write_model_out:
            print("Write model out.")
            joblib.dump(pipe, rf'{out_folder}\rfr_model_{descr}.pkl')
    else:
        print("Load existing model.")
        pipe = joblib.load(existing_model_pth)

    if calc_feature_importance:
        get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)

    if make_accuracy_assessment:
        df_lst = []
        acc_dict = accuracy_assessment(pipe, X_test, y_test, out_folder, descr)
        acc_dict["n_estimators"] = n_estimators
        acc_dict["train_size"] = train_size
        acc_dict["train_num"] = num_train
        acc_dict["test_num"] = num_test

        df = pd.DataFrame.from_dict(acc_dict, orient="index")
        df.columns = [descr]
        df_lst.append(df)

        acc_dict = accuracy_assessment(pipe, X_train, y_train, out_folder, descr + '_in_sample')
        acc_dict["n_estimators"] = n_estimators
        acc_dict["train_size"] = train_size
        acc_dict["test_size"] = test_size
        acc_dict["train_num"] = num_train
        acc_dict["test_num"] = num_test

        df = pd.DataFrame.from_dict(acc_dict, orient="index")
        df.columns = [descr + '_in_sample']
        df_lst.append(df)

        print("Write accuracy results out.")
        df_out = pd.concat(df_lst, axis=1)
        df_out.to_csv(rf"{out_folder}\accuracy_{descr}.csv")

    if make_predictions:
        ## Calculate predictions
        print("Calculate predictions.")
        predictions = pipe.predict(X_encoded)

        iacs["rfr_pred"] = predictions
        iacs["errors"] = iacs["farm_size"] - iacs["rfr_pred"]
        iacs = iacs[["field_id", "farm_size", "rfr_pred", "errors"]].copy()

        print("Write results out.")
        iacs.to_csv(fr"{out_folder}\rfr_predictions_{descr}.csv", index=False)


def out_of_sample_accuracy(model_path, initial_sample_pth, full_data_set_pth, out_folder):

    print("Prepare data.")
    n_estimators = 1000
    train_size = 0.25
    descr = f"comparison_to_best_bayes_model_n{n_estimators}_t{train_size}".replace('.', '')
    pipe = joblib.load(model_path)

    # crop_var = "new_IDKTYP"  ## choose between crop class or functional diverstiy - ID_KTYP or fd or temp_div_agg
    # interaction crop var * state
    # iacs["inter_cr_st"] = iacs[crop_var] + iacs["federal_st"]
    # iacs["inter_sfm_prag"] = iacs["surrf_mean_log"] + iacs["propAg1000"]

    initial_sample = pd.read_csv(initial_sample_pth)
    iacs = pd.read_csv(full_data_set_pth)
    iacs = iacs.loc[iacs["surrf_mean"] > 0].copy()

    ## Calculate natural logarithm
    iacs["log_field_size"] = np.log(iacs["field_size"] * 10000)
    iacs["log_farm_size"] = np.log(iacs["farm_size"] * 10000)
    iacs["surrf_mean_log"] = np.log(iacs["surrf_mean"] * 10000)

    ## Scale
    scale = StandardScaler()
    iacs["log_field_size"] = scale.fit_transform(iacs[["log_field_size"]])
    iacs["surrf_mean_log"] = scale.fit_transform(iacs[["surrf_mean_log"]])
    iacs["propAg1000"] = scale.fit_transform(iacs[["propAg1000"]])
    iacs["SQRAvrg"] = scale.fit_transform(iacs[["SQRAvrg"]])

    ## Drop SQR NAs
    iacs = iacs.loc[iacs["SQRAvrg"].notna()].copy()

    ## Create interaction terms
    iacs["inter_sfm_prag"] = iacs["surrf_mean_log"] + iacs["propAg1000"]
    iacs["inter_cr_st"] = iacs["new_IDKTYP"] + iacs["federal_st"]

    ## test if log is same between R and numpy
    ids = initial_sample["field_id"][:1].tolist()
    for col in ["log_field_size", "log_farm_size", "surrf_mean_log", "propAg1000", "SQRAvrg"]:
        print(col)
        print("\tMax log scaled:", initial_sample.loc[initial_sample["field_id"].isin(ids), col].iloc[0])
        print("\tMy log scaled:", iacs.loc[iacs["field_id"].isin(ids), col].iloc[0])

    dep_var = "log_farm_size"
    indep_vars = ["log_field_size", "surrf_mean_log", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"]

    ## Exclude unnecessary columns
    df_data = iacs[[dep_var] + indep_vars].copy()
    ## Separate dependent and independent variables
    y = df_data[dep_var]
    X = df_data.loc[:, df_data.columns != dep_var]

    ## Convert categorical variables into dummy variables
    X_encoded = pd.get_dummies(
        data=X,
        columns=["inter_cr_st"], drop_first=False)

    ## Calculate predictions
    print("Calculate predictions.")
    X_encoded["predictions"] = pipe.predict(X_encoded)

    ## Get test data set
    indeces = iacs.loc[~iacs["field_id"].isin(initial_sample["field_id"].tolist())]
    indeces = list(indeces.index)

    predictions_log = X_encoded["predictions"].loc[indeces].to_numpy()
    X_test = X_encoded.loc[indeces]
    X_test.drop(columns="predictions", inplace=True)
    y_test = y.loc[indeces]

    print("Calculate accuracy.")
    r2 = pipe.score(X_test, y_test)
    errors_log = abs(predictions_log - y_test)
    mae_log = round(np.mean(errors_log), 2)
    mape = (errors_log / y_test) * 100
    accuracy = 100 - np.mean(mape)

    errors_orig = abs(np.exp(predictions_log) - np.exp(y_test))
    mae_orig = round(np.mean(errors_orig), 2)
    mape_orig = (errors_orig / np.exp(y_test)) * 100
    accuracy_orig = 100 - np.mean(mape_orig)

    acc_dict = {"r2_log": r2,
                "mae_log": mae_log,
                "accuracy_log": accuracy,
                "mae_orig": mae_orig,
                "accuracy_orig": accuracy_orig}

    sns.scatterplot(y=predictions_log, x=y_test, s=1)
    plt.ylabel("Log(Farm size prediction)")
    plt.xlabel("Log(Farm size reference)")
    plt.savefig(f"{out_folder}\scatter_pred_log_vs_ref_log_farm_size_{descr}.png")
    plt.close()

    sns.scatterplot(y=np.exp(predictions_log), x=np.exp(y_test), s=1)
    plt.ylabel("Farm size prediction")
    plt.xlabel("Farm size reference")
    plt.savefig(f"{out_folder}\scatter_pred_orig_vs_ref_orig_farm_size_{descr}.png")
    plt.close()

    acc_dict["n_estimators"] = n_estimators
    acc_dict["train_size"] = train_size
    acc_dict["train_num"] = len(y)
    acc_dict["test_num"] = len(y)

    df = pd.DataFrame.from_dict(acc_dict, orient="index")
    df.columns = [descr]

    print("Write accuracy results out.")
    df.to_csv(fr"{out_folder}\accuracy_{descr}.csv")


def random_points_in_polygons(gpkg_path, n, r, out_pth):
    # Register drivers
    ogr.RegisterAll()

    # Open the gpkg file
    ds = ogr.Open(gpkg_path, 1)  # 1 means read-write mode
    if ds is None:
        raise ValueError("Could not open the GeoPackage.")

    # Assuming there's only one layer in the gpkg
    layer = ds.GetLayerByIndex(0)
    layer_extent = layer.GetExtent()

    generated_points = []

    # Get the number of features in the layer
    feature_count = layer.GetFeatureCount()

    # Generate random points and their buffers with required conditions
    print(f"Generate {n} random points.")
    while len(generated_points) < n:
        # Generate a random feature ID
        random_feature_id = random.randint(0, feature_count - 1)

        # Get the feature by its ID
        random_feature = layer.GetFeature(random_feature_id)
        geom = random_feature.geometry()
        point = geom.Centroid()

        # x = random.uniform(layer_extent[0], layer_extent[1])
        # y = random.uniform(layer_extent[2], layer_extent[3])
        #
        # point = ogr.Geometry(ogr.wkbPoint)
        # point.AddPoint(x, y)
        #
        # # Check if point is inside any polygon
        # layer.SetSpatialFilter(point)
        # if not any(feat for feat in layer):
        #     continue

        too_close = False
        for existing_point in generated_points:
            if existing_point.Distance(point) < 2 * r:
                too_close = True
                break

        if too_close:
            continue

        generated_points.append(point)

    # Find polygons that intersect the buffer of each point
    print(f"Loop over points, buffer and select fields.")
    id_list = []
    for point in generated_points:
        buffer_polygon = point.Buffer(r)
        layer.SetSpatialFilter(buffer_polygon)

        for feat in layer:
            field_id = feat.GetField("field_id")
            if field_id not in id_list:
                id_list.append(field_id)

    # Write the ids to out_pth
    with open(out_pth, 'w') as f:
        for id_ in id_list:
            f.write(str(id_) + '\n')

    ds = None  # Close the datasource


## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    # random_forest_wrapper_old(
    #     iacs_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\all_predictors_sqr.shp",
    #     add_vars_pth=r"Q:\FORLand\Field_farm_size_relationship\data\tables\surrounding_fields_stats_ALL.csv")

    # folder = r"Q:\FORLand\Field_farm_size_relationship\models\rfr_final_model"
    # random_forest_wrapper_new(
    #     iacs_pth=r"Q:\FORLand\Field_farm_size_relationship\data\tables\sample_large_025.csv",
    #     out_folder=folder
    # )

    # out_of_sample_accuracy(
    #     model_path=rf"{folder}\rfr_model__comparison_to_best_bayes_model_n1000_t025.pkl",
    #     initial_sample_pth=r"Q:\FORLand\Field_farm_size_relationship\data\tables\sample_large_025.csv",
    #     full_data_set_pth=rf'Q:\FORLand\Field_farm_size_relationship\data\test_run2\all_predictors_w_grassland_.csv',
    #     out_folder=folder
    # )

    ########################################## Prepare data ##########################################
    # ToDo: Clean the input data by removing all field < 100m² --> Run everything again.
    # When creating the IACS_ALL dataset, drop first all fields < 100m²
    # Calculate statistics of surrounding field sizes, include still all fields!
    # Then get rid of farms that consist of only one field
    # Get also rid of farms that are not from SA, TH and BB but have fields there
    # Then log and scale the data


    # log_and_scale_data(
    #     input_pth=rf'data\test_run2\all_predictors_w_grassland_.csv',
    #     output_pth=rf'models\all_predictors_w_grassland_w_sqr_log_and_scaled.csv'
    # )

    ## Grid search on max_features
    m_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, "sqrt"]
    for m_feat in m_features:
        run_descr = f"rfr_grid_search_m_feat_{m_feat}"
        random_forest_wrapper_clean(
            iacs_pth=rf'models\all_predictors_w_grassland_w_sqr_log_and_scaled.csv',
            dep_var="farm_size_r",
            indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
            categorical_vars=["inter_cr_st"],
            n_estimators=200,
            train_size=0.05,
            test_size=0.05,
            max_features=m_feat,
            run_descr=run_descr,
            out_folder=rf"models\{run_descr}",
            make_predictions=False,
            write_model_out=False
        )

    # run_descr = "rfr_all_variables"
    # print(f"############# {run_descr}")
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_w_sqr_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_std", "log_surrf_min", "log_surrf_max", 'propAg1000',
    #                 'avgTRI1000', 'avgTRI0', 'ElevationA', 'inter_sfm_prag', "federal_st", "new_IDKTYP"],
    #     categorical_vars=["federal_st", "new_IDKTYP"],
    #     n_estimators=1000,
    #     train_size=0.25,
    #     test_size=0.75,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=False
    # )

    # run_descr = "rfr_comp_best_bayes_model_replicated_same_ids"
    # print(f"############# {run_descr}")
    # train_id_df = pd.read_csv(r"data\tables\sample_large_025.csv")
    # train_ids = train_id_df["field_id"].tolist()
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_w_sqr_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_ids=train_ids,
    #     test_size=0.75,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=False
    # )

    # run_descr = "rfr_comp_best_bayes_model_replicated_same_ids_all_variables"
    # print(f"############# {run_descr}")
    # train_id_df = pd.read_csv(r"data\tables\sample_large_025.csv")
    # train_ids = train_id_df["field_id"].tolist()
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_w_sqr_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_std", "log_surrf_min", "log_surrf_max", 'propAg1000',
    #                 'SQRAvrg', 'avgTRI1000', 'avgTRI0', 'ElevationA', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_ids=train_ids,
    #     test_size=0.75,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=False
    # )

    # print("############# rfr_comp_best_bayes_model_replicated")
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.25,
    #     test_size=0.75,
    #     run_descr="comp_best_bayes_model_replicated",
    #     out_folder=r"models\rfr_comp_best_bayes_model_replicated",
    #     existing_model_pth=r"models\rfr_comp_best_bayes_model_replicated\rfr_model_comp_best_bayes_model_replicated_n1000_t025.pkl")

    ####
    # run_descr = "comp_best_bayes_model_replicated-05_05"
    # print(f"############# {run_descr}")
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.05,
    #     test_size=0.05,
    #     run_descr=run_descr,
    #     out_folder=fr"models\{run_descr}")

    # Sample fields with random point buffers
    ## Was actually done in QGIS because it was much faster!
    ## Generatr 1500 point with "Zufällige Punkte in den Layergrenzen...", input: Hexagon grid -->
    ## Puffer 1000m, Input: random_point ---> "Nach Position selektieren", Input Puffer, IACS -->
    ## Export - "Gewählte Objekte speichern als" CSV, input IACS

    # random_points_in_polygons(
    #     gpkg_path=r"data\vector\IACS\IACS_ALL_2018_with_grassland_recl_3035.shp",
    #     n=1000,
    #     r=500,
    #     out_pth=r"models\spatial_buffer_sampling_n1000_r500_ogr.csv")

    # run_descr = "comp_best_bayes_model_replicated-05_05_buffer_sampling"
    # print(f"############# {run_descr}")
    # train_id_df = pd.read_csv("Q:\FORLand\Field_farm_size_relationship\models\spatial_buffer_sampling_3000points_1000m.csv")
    # train_ids = train_id_df["field_id"].tolist()
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_ids=train_ids,
    #     train_size=0.05,
    #     test_size=0.05,
    #     run_descr=run_descr,
    #     out_folder=fr"models\{run_descr}",
    #     make_predictions=False,
    #     write_model_out=False
    # )

    # ToDo:
    # Merge smaller surrounding field sizes to variable df
    # Log and scale all variables again
    # Test smaller surrounding field sizes
    # Test all variables

    ####
    # run_descr = "crfr_comp_best_bayes_model_replicated_all_variables-05_05"
    # print(f"############# {run_descr}")
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'models\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_medi", "log_surrf_std", "log_surrf_min",
    #                 "log_surrf_max", "surrf_no_f", "ElevationA", "avgTRI0", "avgTRI1000", "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.05,
    #     test_size=0.05,
    #     run_descr=run_descr,
    #     out_folder=fr"models\{run_descr}")

    # print("############# rfr_comp_best_bayes_model_replicated_all_variables")
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'Q:\FORLand\Field_farm_size_relationship\models\rfr_comp_best_bayes_model_replicated\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_medi", "log_surrf_std", "log_surrf_min",
    #                 "log_surrf_max", "surrf_no_f", "ElevationA", "avgTRI0", "avgTRI1000", "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.25,
    #     test_size=0.75,
    #     run_descr="all_variables",
    #     out_folder=r"Q:\FORLand\Field_farm_size_relationship\models\rfr_comp_best_bayes_model_replicated_all_variables")

    # print("############# rfr_comp_best_bayes_model_repl_50-50")
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'Q:\FORLand\Field_farm_size_relationship\models\rfr_comp_best_bayes_model_replicated\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="log_farm_size",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.5,
    #     test_size=0.5,
    #     run_descr="comp_best_bayes_model_repl_50-50",
    #     out_folder=r"Q:\FORLand\Field_farm_size_relationship\models\rfr_comp_best_bayes_model_repl_50-50")

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
