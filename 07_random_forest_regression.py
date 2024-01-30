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
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib
from sklearn.inspection import PartialDependenceDisplay
# import shap

import numpy as np

# project library for plotting
import plotting_lib
## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"
IACS_PTH = r"data\test_run2\all_predictors\all_predictors.shp"
SAMPLE_PTH = r"data\vector\final\matched_sample-v2.shp"

## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#


def train_random_forest_regressor(X, y, n_estimators, scaler=False, max_features="sqrt", max_depth=8,
                                  criterion="squared_error", n_jobs=2):

    print("Fit regressor")

    ## Design pipeline to build the treatment estimator. It standardizes the data and applies a logistic classifier
    if scaler:
        pipe = Pipeline([
            ('scaler', StandardScaler()),  # alternative is min-max scaler
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=42,
                                                max_depth=max_depth, criterion=criterion, n_jobs=n_jobs))
        ])
    else:
        pipe = Pipeline([
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=42,
                                                max_depth=max_depth, criterion=criterion, n_jobs=n_jobs))
        ])

    ## Fit the classifier to the data
    pipe.fit(X, y)

    return pipe

def get_feature_importances(pipe, X_test, y_test, out_folder, descr):

    print("Determine feature importance.")
    ## Get feature importance
    ## https://mljar.com/blog/feature-importance-in-random-forest/
    sorted_idx = pipe["regressor"].feature_importances_.argsort()
    plt.figure(figsize=(16, 16))
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


def random_forest_wrapper_grid_search(iacs_pth, dep_var, indep_vars, categorical_vars, run_descr, out_folder,
                                      train_size=0.7, test_size=0.3, make_predictions=True,
                                      calc_feature_importance=True, make_accuracy_assessment=True):

    print("Read data.")
    iacs = pd.read_csv(iacs_pth)

    ## Create output folder
    try:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    except OSError:
        print('Error: Creating directory. ' + out_folder)

    ## Define Random Forest parameters
    print("RFR parameters - grid search", "training size:", train_size)
    descr = f"{run_descr}_t{train_size}".replace('.', '')

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
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, train_size=train_size,
                                                    random_state=42)
    num_train = len(y_train)
    num_test = len(y_test)

    df_train = iacs.loc[X_train.index]
    df_train[["field_id"]].to_csv(rf'{out_folder}\training_sample_field_ids.csv', index=False)

    rfr = RandomForestRegressor(random_state=42, n_jobs=10)

    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_features': [1, 2, 3, 4, 5, 6, "sqrt"],
        'max_depth': [1, 2, 4, 6, 8],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
    }

    param_grid = {
        'n_estimators': [1000],
        'max_features': [6],
        'max_depth': [8],
        'criterion': ['squared_error']
    }
    # with open(fr"{out_folder}\paremter_grid.json", 'w') as fp:
    #     json.dump(param_grid, fp)

    ## Do grid search an fit regressor
    print("Grid search.")
    CV_rfr = GridSearchCV(rfr, param_grid)
    CV_rfr.fit(X_train, y_train)

    best_parameters = CV_rfr.best_params_
    with open(fr"{out_folder}\best_parameters.json", 'w') as fp:
        json.dump(best_parameters, fp)

    print(best_parameters)

    pipe = Pipeline([
        ('regressor', RandomForestRegressor(
            n_estimators=best_parameters["n_estimators"],
            max_features=best_parameters["max_features"],
            max_depth=best_parameters["max_depth"],
            criterion=best_parameters["criterion"],
            n_jobs=10
        ))
    ])

    pipe.fit(X_train, y_train)

    if calc_feature_importance:
        get_feature_importances(pipe=pipe, X_test=X_test, y_test=y_test, out_folder=out_folder, descr=descr)

    if make_accuracy_assessment:
        df_lst = []
        acc_dict = accuracy_assessment(pipe, X_test, y_test, out_folder, descr)
        acc_dict["n_estimators"] = best_parameters["n_estimators"]
        acc_dict["max_features"] = best_parameters["max_features"]
        acc_dict["max_depth"] = best_parameters["max_depth"]
        acc_dict["criterion"] = best_parameters["criterion"]
        acc_dict["train_size"] = train_size
        acc_dict["train_num"] = num_train
        acc_dict["test_num"] = num_test

        df = pd.DataFrame.from_dict(acc_dict, orient="index")
        df.columns = [descr]
        df_lst.append(df)

        acc_dict = accuracy_assessment(pipe, X_train, y_train, out_folder, descr + '_in_sample')
        acc_dict["n_estimators"] = best_parameters["n_estimators"]
        acc_dict["max_features"] = best_parameters["max_features"]
        acc_dict["max_depth"] = best_parameters["max_depth"]
        acc_dict["criterion"] = best_parameters["criterion"]
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


def random_forest_wrapper_clean(iacs_pth, dep_var, indep_vars, categorical_vars, run_descr, out_folder, train_ids=None,
                                test_ids=None, n_estimators=500, train_size=0.7, test_size=0.3, max_features="sqrt",
                                criterion='squared_error', max_depth=8,
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

    iacs.dropna(subset=[dep_var] + indep_vars, inplace=True)

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
            print(f"Using the provided {len(test_ids)} for testing.")
            indeces = iacs.loc[iacs["field_id"].isin(test_ids)]
            indeces = list(indeces.index)
            X_test = X_encoded.loc[indeces]
            y_test = y.loc[indeces]
        else:
            indeces = iacs.loc[~iacs["field_id"].isin(train_ids)]
            indeces = list(indeces.index)
            # random_indeces = random.sample(indeces, int(test_size * len(indeces)))
            # X_test = X_encoded.loc[random_indeces]
            # y_test = y.loc[random_indeces]
            X_test = X_encoded.loc[indeces]
            y_test = y.loc[indeces]
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
        pipe = train_random_forest_regressor(X_train, y_train, n_estimators=n_estimators, scaler=False,
                                             max_features=max_features, max_depth=max_depth, criterion=criterion,
                                             n_jobs=10)
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
        acc_dict["max_features"] = max_features
        acc_dict["max_depth"] = max_depth
        acc_dict["criterion"] = criterion
        acc_dict["train_size"] = train_size
        acc_dict["train_num"] = num_train
        acc_dict["test_num"] = num_test

        df = pd.DataFrame.from_dict(acc_dict, orient="index")
        df.columns = [descr]
        df_lst.append(df)

        acc_dict = accuracy_assessment(pipe, X_train, y_train, out_folder, descr + '_in_sample')
        acc_dict["n_estimators"] = n_estimators
        acc_dict["max_features"] = max_features
        acc_dict["max_depth"] = max_depth
        acc_dict["criterion"] = criterion
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
        print(f"Calculate predictions for {len(X_encoded)} cases.")
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


def create_partial_dependence_df(X_train, target_expl_var):

    X_copy = X_train.copy()
    cols = list(X_copy.columns)
    cols.remove(target_expl_var)

    for col in cols:
        X_copy[col] = X_train[col].mean()

    return X_copy
## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    # run_descr = f"rfr_grid_search_correct"
    # random_forest_wrapper_grid_search(
    #     iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     train_size=0.7,
    #     test_size=0.3,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=False
    # )

    ## WITHOUT SQR
    run_descr = f"rfr_final_run_selected_variables_wo_sqr"
    # train_id_df = pd.read_csv(rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv')
    # train_ids = train_id_df["field_id"].tolist()
    field_ids = pd.read_csv(rf"data\tables\predictors\all_field_ids_w_sample.csv")
    train_ids = field_ids.loc[field_ids["matched_sample"] == 1, "field_id"].tolist()
    test_ids = field_ids.loc[field_ids["matched_sample"] == 0, "field_id"].tolist()
    random_forest_wrapper_clean(
        iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
        dep_var="farm_size_r",
        indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'inter_sfm_prag', "inter_cr_st"],
        categorical_vars=["inter_cr_st"],
        n_estimators=1000,
        train_ids=train_ids,
        test_ids=test_ids,
        max_features=6,
        criterion="squared_error",
        max_depth=8,
        run_descr=run_descr,
        out_folder=rf"models\{run_descr}",
        make_predictions=True,
        write_model_out=False
    )

    run_descr = f"rfr_final_run_all_variables_wo_sqr"
    # train_id_df = pd.read_csv(rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv')
    # train_ids = train_id_df["field_id"].tolist()
    random_forest_wrapper_clean(
        iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
        dep_var="farm_size_r",
        indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_std", "log_surrf_min", "log_surrf_max", 'propAg1000',
                    'avgTRI1000', 'avgTRI0', 'ElevationA', 'inter_sfm_prag', "inter_cr_st"],
        categorical_vars=["inter_cr_st"],
        n_estimators=1000,
        train_ids=train_ids,
        test_ids=test_ids,
        max_features=6,
        criterion="squared_error",
        max_depth=8,
        run_descr=run_descr,
        out_folder=rf"models\{run_descr}",
        make_predictions=True,
        write_model_out=True
    )

    ## WITH SQR
    # run_descr = f"rfr_final_run_selected_variables"
    # # train_id_df = pd.read_csv(rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv')
    # # train_ids = train_id_df["field_id"].tolist()
    # field_ids = pd.read_csv(rf"data\tables\predictors\all_field_ids_w_sample.csv")
    # train_ids = field_ids.loc[field_ids["matched_sample"] == 1, "field_id"].tolist()
    # test_ids = field_ids.loc[field_ids["matched_sample"] == 0, "field_id"].tolist()
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_ids=train_ids,
    #     test_ids=test_ids,
    #     max_features=6,
    #     criterion="squared_error",
    #     max_depth=8,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=False
    # )
    #
    # run_descr = f"rfr_final_run_all_variables"
    # # train_id_df = pd.read_csv(rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv')
    # # train_ids = train_id_df["field_id"].tolist()
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_std", "log_surrf_min", "log_surrf_max", 'propAg1000',
    #                 'SQRAvrg', 'avgTRI1000', 'avgTRI0', 'ElevationA', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_ids=train_ids,
    #     test_ids=test_ids,
    #     max_features=6,
    #     criterion="squared_error",
    #     max_depth=8,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=True
    # )

    # run_descr = f"rfr_test_run_all_variables"
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", "log_surrf_std", "log_surrf_min", "log_surrf_max", 'propAg1000',
    #                 'SQRAvrg', 'avgTRI1000', 'avgTRI0', 'ElevationA', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.01,
    #     max_features=6,
    #     criterion="squared_error",
    #     max_depth=8,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=True
    # )

    # run_descr = f"rfr_test_run_selected_variables"
    # random_forest_wrapper_clean(
    #     iacs_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
    #     dep_var="farm_size_r",
    #     indep_vars=["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"],
    #     categorical_vars=["inter_cr_st"],
    #     n_estimators=1000,
    #     train_size=0.01,
    #     max_features=6,
    #     criterion="squared_error",
    #     max_depth=8,
    #     run_descr=run_descr,
    #     out_folder=rf"models\{run_descr}",
    #     make_predictions=True,
    #     write_model_out=True
    # )

    # ##### PLOT PARTIAL DEPENDENCE PLOTS
    # ## Define Inuput
    # # run_descr = f"rfr_test_run_all_variables"
    # run_descr = f"rfr_test_run_selected_variables"
    # # train_id_df = pd.read_csv(rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv')
    # train_ids_pth = rf'models\{run_descr}\training_sample_field_ids.csv'
    # iacs_pth = rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv'
    # existing_model_pth = rf"models\{run_descr}\rfr_model_{run_descr}_n1000_t001.pkl"
    # dep_var = "farm_size_r"
    # # indep_vars = ["log_field_size", "log_surrf_mean", "log_surrf_std", "log_surrf_min", "log_surrf_max", 'propAg1000',
    # #               'SQRAvrg', 'avgTRI1000', 'avgTRI0', 'ElevationA', 'inter_sfm_prag', "inter_cr_st"]
    #
    # indep_vars = ["log_field_size", "log_surrf_mean", 'propAg1000', 'SQRAvrg', 'inter_sfm_prag', "inter_cr_st"]
    # continous_cols = ['log_field_size', 'log_surrf_mean', 'propAg1000', 'SQRAvrg', 'inter_sfm_prag']
    # # 'log_surrf_std', 'log_surrf_min', 'log_surrf_max', 'avgTRI1000', 'avgTRI0', 'ElevationA',
    # categorical_vars = ["inter_cr_st"]
    # test_size = 0.3
    #
    # ## Load data
    # print("Load data.")
    # iacs = pd.read_csv(iacs_pth)
    # train_id_df = pd.read_csv(train_ids_pth)
    # pipe = joblib.load(existing_model_pth)
    #
    # ## Prepare data
    # print("Prepare data.")
    # train_ids = train_id_df["field_id"].tolist()
    #
    # df_data = iacs[[dep_var] + indep_vars].copy()
    # y = df_data[dep_var]
    # X = df_data.loc[:, df_data.columns != dep_var]
    #
    # X_encoded = pd.get_dummies(
    #     data=X,
    #     columns=categorical_vars,
    #     drop_first=False)
    #
    # indeces = iacs.loc[iacs["field_id"].isin(train_ids)]
    # indeces = list(indeces.index)
    # X_train = X_encoded.loc[indeces]
    # # y_train = y.loc[indeces]
    # # indeces = iacs.loc[~iacs["field_id"].isin(train_ids)]
    # # indeces = list(indeces.index)
    # # random_indeces = random.sample(indeces, int(test_size * len(indeces)))
    # # X_test = X_encoded.loc[random_indeces]
    # # y_test = y.loc[random_indeces]
    #
    # target_expl_var = "log_field_size"
    # # X_pd = create_partial_dependence_df(X_train, target_expl_var)
    # # predictions = pipe.predict(X_pd)
    #
    # # X_pd["rfr_pred"] = predictions
    # # X_pd.sort_values(target_expl_var)
    #
    # # new_col = target_expl_var.replace("log_", "")
    # # X_pd[new_col] = np.exp(X_pd[target_expl_var])
    #
    # # fig, axs = plt.subplots()
    # # sns.lineplot(data=X_pd, x=new_col, y="rfr_pred", ax=axs)
    # # fig.tight_layout()
    # # plt.savefig(rf"models\{run_descr}\partial_dependence_test_manual.png", dpi=300)
    # # plt.close()
    #
    # color_dict = {
    #     "BB": "#1f77b4",
    #     "SA": "#ff7f0e",
    #     "TH": "#2ca02c",
    #     "BV": "#d62728",
    #     "LS": "#9467bd"
    # }
    #
    # # vars = ["log_surrf_std", "log_surrf_mean", "log_surrf_max", "inter_sfm_prag"]
    # vars = ["log_field_size", "log_surrf_mean", "propAg1000", "SQRAvrg", "inter_sfm_prag"]
    # for target_expl_var in vars:
    #
    #     states = ["BB", "SA", "TH"]
    #     crops = ["CE", "GR", "LE", "MA", "OR", "OT", "PO", "SU"]
    #
    #     print("Plot partial dependence plot.")
    #     fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=plotting_lib.cm2inch(20, 20))
    #
    #     binary_cols = list(X_train.columns)
    #     for col in continous_cols:
    #         binary_cols.remove(col)
    #
    #     for i, crop in enumerate(crops):
    #
    #         ix = np.unravel_index(i, axs.shape)
    #         for state in states:
    #             curr_bin_col = f"inter_cr_st_{crop}{state}"
    #             X_copy = X_train.copy()
    #
    #             cols = continous_cols.copy()
    #             cols.remove(target_expl_var)
    #             for col in cols:
    #                 X_copy[col] = X_train[col].mean()
    #
    #             cols = binary_cols.copy()
    #             cols.remove(curr_bin_col)
    #             for col in cols:
    #                 X_copy[col] = 0
    #             X_copy[curr_bin_col] = 1
    #
    #             predictions = pipe.predict(X_copy)
    #
    #             X_copy["rfr_pred"] = predictions
    #             X_copy.sort_values(target_expl_var)
    #
    #             if "log_" in target_expl_var:
    #                 new_col = target_expl_var.replace("log_", "")
    #                 X_copy[new_col] = np.exp(X_copy[target_expl_var])
    #             else:
    #                 new_col = target_expl_var
    #
    #             sns.lineplot(data=X_copy, x=new_col, y="rfr_pred", ax=axs[ix], color=color_dict[state])
    #         axs[ix].set_title(crop)
    #
    #     ix = np.unravel_index(i+1, axs.shape)
    #     legend_elements = [Patch(facecolor=color_dict[state], edgecolor=None, label=state) for state in states]
    #     axs[ix].legend(handles=legend_elements, ncol=1, title=None)
    #
    #     fig.tight_layout()
    #     plt.savefig(rf"models\{run_descr}\partial_dependence_manual_{target_expl_var}.png", dpi=300)
    #     plt.close()
    #
    # # p = PartialDependenceDisplay.from_estimator(pipe, X_train, range(6), n_jobs=12)
    # # plt.tight_layout()
    # # plt.savefig(rf"models\{run_descr}\partial_dependence_test.png", dpi=300)
    # # plt.close()

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
