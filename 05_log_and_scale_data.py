## Authors:
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"


def log_and_scale_data(input_pth, output_pth):
    print("Read Input data.")
    iacs = pd.read_csv(input_pth, dtype={"farm_id": str})

    print("Prepare data.")

    ## Subtract field from farm size
    iacs["farm_size_r"] = iacs["farm_size"] - iacs["field_size"]

    ## Due to some different calculations of the field sizes in prior steps (R-pyhton), sometimes the field sizes are
    ## only slightly smaller than the farm sizes, causing trouble in later steps
    iacs.loc[iacs["fieldCount"] == 1, "farm_size_r"] = 0.001
    iacs.loc[iacs["farm_size_r"] == 0, "farm_size_r"] = 0.001
    iacs.loc[iacs["fieldCount"] == 1, "farm_size"] = iacs.loc[iacs["fieldCount"] == 1, "field_size"]

    print("min. farm_size_r:", iacs["farm_size_r"].min(), "\tmin. farm_size", iacs["farm_size_r"].min())

    ## Calculate values to m²
    cols = ["field_size", "surrf_mean", "surrf_std", "surrf_min", "surrf_max"]
    for col in cols:
        iacs[col] = iacs[col] * 10000

    ## Remove all fields smaller 100m²
    iacs = iacs.loc[iacs["field_size"] >= 100].copy()

    ## Drop zeros in the surrounding field statistics
    cols = ["surrf_mean", "surrf_std", "surrf_min", "surrf_max"]
    for col in cols:
        iacs = iacs.loc[iacs[col] > 0].copy()

    ## Calculate natural logarithm
    cols = ["field_size", "farm_size", "farm_size_r", "surrf_mean", "surrf_std", "surrf_min", "surrf_max"]
    for col in cols:
        iacs[f"log_{col}"] = np.log(iacs[col])
        iacs[col] = round(iacs[col], 4)
        iacs[f"log_{col}"] = round(iacs[f"log_{col}"], 4)
    log_cols = [f"log_{col}" for col in cols]

    ## Create interaction terms
    iacs["inter_cr_st"] = iacs["new_ID_KTYP"] + iacs["federal_st"]
    iacs["inter_sfm_prag"] = iacs["log_surrf_mean"] + iacs["propAg1000"]

    ## Scale variables
    scale_cols = log_cols + ["inter_sfm_prag", "propAg1000", "ElevationA", "sdElev0", "avgTRI0", "avgTRI500",
                  "avgTRI1000", "SQRAvrg"]
    scale_cols.remove("log_farm_size")
    scale_cols.remove("log_farm_size_r")
    scale = StandardScaler()
    for col in scale_cols:
        iacs[col] = scale.fit_transform(iacs[[col]])
        iacs[col] = round(iacs[col], 4)


    ## Drop farms that have likely most of their areas outside our study area
    iacs["fstate_id"] = iacs["farm_id"].apply(lambda x: x[:2])
    drop_fstate_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13", "14"]
    iacs = iacs.loc[~(iacs["fstate_id"].isin(drop_fstate_ids) & iacs["federal_st"].isin(["BB", "SA", "TH"]))].copy()
    print("Remaining observations:", len(iacs))

    print("Write out")
    iacs.to_csv(output_pth, index=False)


def subset_training_data(log_and_scaled_data_pth, sample_pth, output_pth):
    # Subset training data
    print("Read input.")
    df = pd.read_csv(log_and_scaled_data_pth)
    df_ids = pd.read_csv(sample_pth)

    print("Get sample.")
    df = pd.merge(df, df_ids, how="left", on="field_id")
    df = df.loc[df["matched_sample"] == 1].copy()

    print("Write ouput.")
    df.to_csv(output_pth)


## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    log_and_scale_data(
        input_pth=rf'data\tables\predictors\all_predictors_w_grassland.csv',
        output_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv'
    )

    subset_training_data(
        log_and_scaled_data_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled.csv',
        sample_pth=rf"data\tables\predictors\all_field_ids_w_sample.csv",
        output_pth=rf'data\tables\predictors\all_predictors_w_grassland_log_and_scaled_matched_sample.csv'
    )

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
